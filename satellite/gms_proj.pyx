# cython: language_level=3
from libc.math cimport sqrt, M_PI, atan2, sin, cos, floor
cimport cython

from collections import namedtuple

import numpy as np
cimport numpy as cnp

cnp.import_array()

cdef double EARTH_FLATTENING = 1 / 298.257
cdef double EARTH_EQUATORIAL_RADIUS = 6378136.0
cdef double EARTH_POLAR_RADIUS = EARTH_EQUATORIAL_RADIUS * (1 - EARTH_FLATTENING)

cdef struct Pixel:
    int line
    int pixel

cdef struct Vector2D:
    double x
    double y

cdef struct Vector3D:
    double x
    double y
    double z

cdef struct Satpos:
    double[18] x
    double[18] y
    double[18] z

cdef struct Attitude:
    double[33] angle_between_earth_and_sun
    double[33] angle_between_sat_spin_and_z_axis
    double[33] angle_between_sat_spin_and_yz_plane

cdef struct OrbitAngles:
    double[18] greenwich_sidereal_time
    double[18] declination_from_sat_to_sun
    double[18] right_ascension_from_sat_to_sun

cdef struct Orbit:
    OrbitAngles angles
    Satpos sat_position
    double[3][3] nutation_precession
"""Orbital Parameters

Args:
    angles (OrbitAngles): Orbit angles
    sat_position (Vector3D): Satellite position
    nutation_precession: Nutation and precession matrix (3x3)
"""

cdef struct ScanningParameters:
    double start_time_of_scan
    double spinning_rate
    int num_sensors
    double sampling_angle

cdef struct ImageOffset:
    double line_offset
    double pixel_offset

cdef struct ScanningAngles:
    double stepping_angle
    double sampling_angle
    double[3][3] misalignment

"""Scanning angles

Args:
    stepping_angle: Scanning angle along line (rad)
    sampling_angle: Scanning angle along pixel (rad)
    misalignment: Misalignment matrix (3x3)
"""

cdef struct EarthEllipsoid:
    double flattening
    double equatorial_radius

cdef struct _AttitudePrediction:
    double prediction_times
    Attitude attitude

cdef struct ProjectionParameters:
    ImageOffset image_offset
    ScanningAngles scanning_angles
    EarthEllipsoid earth_ellipsoid
"""Projection parameters.

Args:
    image_offset (ImageOffset): Image offset
    scanning_angles (ScanningAngles): Scanning angles
    earth_ellipsoid (EarthEllipsoid): Earth ellipsoid
"""

cdef struct _OrbitPrediction:
    double prediction_times
    OrbitAngles angles
    Satpos sat_position
    double[3][3] nutation_precession

cdef struct PredictedNavigationParameters:
    _AttitudePrediction attitude
    _OrbitPrediction orbit
"""Predictions of time-dependent navigation parameters.

They need to be evaluated for each pixel.

Args:
    attitude (AttitudePrediction): Attitude prediction
    orbit (OrbitPrediction): Orbit prediction
"""

cdef struct PixelNavigationParameters:
    Attitude attitude
    Orbit orbit
    ProjectionParameters proj_params
"""Navigation parameters for a single pixel.

Args:
    attitude (Attitude): Attitude parameters
    orbit (Orbit): Orbit parameters
    proj_params (ProjectionParameters): Projection parameters
"""


cdef struct StaticNavigationParameters:
    ProjectionParameters proj_params
    ScanningParameters scan_params
"""Navigation parameters which are constant for the entire scan.

Args:
    proj_params (ProjectionParameters): Projection parameters
    scan_params (ScanningParameters): Scanning parameters
"""

cdef struct ImageNavigationParameters:
    StaticNavigationParameters static_
    PredictedNavigationParameters predicted
"""Navigation parameters for the entire image.

Args:
    static (StaticNavigationParameters): Static parameters.
    predicted (PredictedNavigationParameters): Predicted time-dependent parameters.
"""

cdef class AttitudePrediction:
    """Attitude prediction.

    Use .to_numba() to pass this object to jitted methods. This extra
    layer avoids usage of jitclasses and having to re-implement np.unwrap in
    numba.
    """
    cdef double prediction_times
    cdef Attitude attitude

    def __init__(
        self,
        double prediction_times,
        Attitude attitude
    ):
        """Initialize attitude prediction.

        In order to accelerate interpolation, the 2-pi periodicity of angles
        is unwrapped here already (that means phase jumps greater than pi
        are wrapped to their 2*pi complement).

        Args:
            prediction_times: Timestamps of predicted attitudes
            attitude (Attitude): Attitudes at prediction times
        """
        self.prediction_times = prediction_times
        self.attitude = self._unwrap_angles(attitude)

    cdef Attitude _unwrap_angles(self, Attitude attitude):
        # TODO: Implement unwrap
        return Attitude(
            np.unwrap([attitude.angle_between_earth_and_sun])[0],
            np.unwrap([attitude.angle_between_sat_spin_and_z_axis])[0],
            np.unwrap([attitude.angle_between_sat_spin_and_yz_plane])[0],
        )

    def to_numba(self):
        """Convert to numba-compatible type."""
        return _AttitudePrediction(
            prediction_times=self.prediction_times,
            attitude=self.attitude
        )


cdef class OrbitPrediction:
    """Orbit prediction.

    Use .to_numba() to pass this object to jitted methods. This extra
    layer avoids usage of jitclasses and having to re-implement np.unwrap in
    numba.
    """
    cdef double prediction_times
    cdef OrbitAngles angles
    cdef Satpos sat_position
    cdef double[3][3] nutation_precession

    @cython.boundscheck(False)
    def __init__(
        self,
        double prediction_times,
        OrbitAngles angles,
        Satpos sat_position,
        double[:, :] nutation_precession,
    ):
        """Initialize orbit prediction.

        In order to accelerate interpolation, the 2-pi periodicity of angles
        is unwrapped here already (that means phase jumps greater than pi
        are wrapped to their 2*pi complement).

        Args:
            prediction_times: Timestamps of orbit prediction.
            angles (OrbitAngles): Orbit angles
            sat_position (Vector3D): Satellite position
            nutation_precession: Nutation and precession matrix.
        """
        self.prediction_times = prediction_times
        self.angles = self._unwrap_angles(angles)
        self.sat_position = sat_position
        cdef int i, j
        for i in range(3):
            for j in range(3):
                self.nutation_precession[i][j] = nutation_precession[i][j]

    def _unwrap_angles(self, angles: OrbitAngles):
        cdef OrbitAngles o
        o.greenwich_sidereal_time=np.unwrap(angles.greenwich_sidereal_time)
        o.declination_from_sat_to_sun=np.unwrap(angles.declination_from_sat_to_sun)
        o.right_ascension_from_sat_to_sun=np.unwrap(
                angles.right_ascension_from_sat_to_sun
            )
        return o

    def to_numba(self):
        """Convert to numba-compatible type."""
        return _OrbitPrediction(
            prediction_times=self.prediction_times,
            angles=self.angles,
            sat_position=self.sat_position,
            nutation_precession=self.nutation_precession,
        )

cdef get_lons_lats(lines, pixels, ImageNavigationParameters nav_params):
    """Compute lon/lat coordinates given VISSR image coordinates.

    Args:
        lines: VISSR image lines
        pixels: VISSR image pixels
        nav_params: Image navigation parameters
    """
    pixels_2d, lines_2d = np.meshgrid(pixels, lines)
    lons, lats = _get_lons_lats_numba(lines_2d, pixels_2d, nav_params=nav_params)
    return lons, lats


cdef _get_lons_lats_numba(lines_2d, pixels_2d, ImageNavigationParameters nav_params):
    cdef int shape_x, shape_y
    cdef double[:, :] lons, lats
    shape_x, shape_y = lines_2d.shape
    lons = np.zeros(lines_2d.shape, dtype=np.float32)
    lats = np.zeros(lines_2d.shape, dtype=np.float32)
    for i in range(shape_x):
        for j in range(shape_y):
            pixel = Pixel(lines_2d[i, j], pixels_2d[i, j])
            nav_params_pix = _get_pixel_navigation_parameters(
                pixel, nav_params
            )
            lon, lat = get_lon_lat(pixel, nav_params_pix)
            lons[i, j] = lon
            lats[i, j] = lat
    # Stack lons and lats because da.map_blocks doesn't support multiple
    # return values.
    return np.stack((lons, lats))

cdef PixelNavigationParameters _get_pixel_navigation_parameters(point, ImageNavigationParameters im_nav_params):
    obs_time = get_observation_time(point, im_nav_params.static_.scan_params)
    attitude, orbit = interpolate_navigation_prediction(
        attitude_prediction=im_nav_params.predicted.attitude,
        orbit_prediction=im_nav_params.predicted.orbit,
        observation_time=obs_time
    )
    return PixelNavigationParameters(
        attitude=attitude,
        orbit=orbit,
        proj_params=im_nav_params.static_.proj_params
    )


cdef double get_observation_time(point, scan_params):
    """Calculate observation time of a VISSR pixel."""
    relative_time = _get_relative_observation_time(point, scan_params)
    return scan_params.start_time_of_scan + relative_time

@cython.cdivision(True)
cdef double _get_relative_observation_time(point, ScanningParameters scan_params):
    cdef int line, pixel
    line, pixel = point
    pixel = pixel + 1
    line = line + 1
    spinning_freq = 1440 * scan_params.spinning_rate
    line_step = floor((line - 1) / scan_params.num_sensors)
    pixel_step = (scan_params.sampling_angle * pixel) / (2 * M_PI)
    return (line_step + pixel_step) / spinning_freq

cdef interpolate_navigation_prediction(
    _AttitudePrediction attitude_prediction, _OrbitPrediction orbit_prediction, double observation_time
):
    """Interpolate predicted navigation parameters."""
    attitude = interpolate_attitude_prediction(attitude_prediction, observation_time)
    orbit = interpolate_orbit_prediction(orbit_prediction, observation_time)
    return attitude, orbit


cdef get_lon_lat(Pixel pixel, PixelNavigationParameters nav_params):
    """Get longitude and latitude coordinates for a given image pixel.

    Args:
        pixel (Pixel): Point in image coordinates.
        nav_params (PixelNavigationParameters): Navigation parameters for a
            single pixel.

    Returns:
        Longitude and latitude in degrees.
    """
    scan_angles = transform_image_coords_to_scanning_angles(
        pixel,
        nav_params.proj_params.image_offset,
        nav_params.proj_params.scanning_angles
    )
    view_vector_sat = transform_scanning_angles_to_satellite_coords(
        scan_angles,
        nav_params.proj_params.scanning_angles.misalignment
    )
    view_vector_earth_fixed = transform_satellite_to_earth_fixed_coords(
        view_vector_sat,
        nav_params.orbit,
        nav_params.attitude
    )
    point_on_earth = intersect_with_earth(
        view_vector_earth_fixed,
        nav_params.orbit.sat_position,
        nav_params.proj_params.earth_ellipsoid
    )
    lon, lat = transform_earth_fixed_to_geodetic_coords(
        point_on_earth, nav_params.proj_params.earth_ellipsoid.flattening
    )
    return lon, lat

cdef Vector2D transform_image_coords_to_scanning_angles(Pixel point, ImageOffset image_offset, ScanningAngles scanning_angles):
    """Transform image coordinates to scanning angles.

    Args:
        point (Pixel): Point in image coordinates.
        image_offset (ImageOffset): Image offset.
        scanning_angles (ScanningAngles): Scanning angles.

    Returns:
        Scanning angles (x, y) at the pixel center (rad).
    """
    line_offset = image_offset.line_offset
    pixel_offset = image_offset.pixel_offset
    stepping_angle = scanning_angles.stepping_angle
    sampling_angle = scanning_angles.sampling_angle
    x = sampling_angle * (point.pixel + 1 - pixel_offset)
    y = stepping_angle * (point.line + 1 - line_offset)
    return Vector2D(x, y)


cdef Vector3D transform_scanning_angles_to_satellite_coords(Vector2D angles, double[:, :] misalignment):
    """Transform scanning angles to satellite angular momentum coordinates.

    Args:
        angles (Vector2D): Scanning angles in radians.
        misalignment: Misalignment matrix (3x3)

    Returns:
        View vector (Vector3D) in satellite angular momentum coordinates.
    """
    x, y = angles.x, angles.y
    sin_x = sin(x)
    cos_x = cos(x)
    view = Vector3D(cos(y), 0.0, sin(y))

    # Correct for misalignment
    view = matrix_vector(misalignment, view)

    # Rotate around z-axis
    return Vector3D(
        cos_x * view.x - sin_x * view.y,
        sin_x * view.x + cos_x * view.y,
        view.z
    )


cdef Vector3D transform_satellite_to_earth_fixed_coords(
    Vector3D point,
    Orbit orbit,
    Attitude attitude
):
    """Transform from earth-fixed to satellite angular momentum coordinates.

    Args:
        point (Vector3D): Point in satellite angular momentum coordinates.
        orbit (Orbit): Orbital parameters
        attitude (Attitude): Attitude parameters
    Returns:
        Point (Vector3D) in earth-fixed coordinates.
    """
    unit_vector_z = _get_satellite_unit_vector_z(attitude, orbit)
    unit_vector_x = _get_satellite_unit_vector_x(unit_vector_z, attitude, orbit)
    unit_vector_y = _get_satellite_unit_vector_y(unit_vector_x, unit_vector_z)
    return _get_earth_fixed_coords(
        point,
        unit_vector_x,
        unit_vector_y,
        unit_vector_z
    )


cdef Vector3D _get_satellite_unit_vector_z(Attitude attitude, Orbit orbit):
    v1950 = _get_satellite_z_axis_1950(
        attitude.angle_between_sat_spin_and_z_axis,
        attitude.angle_between_sat_spin_and_yz_plane
    )
    vcorr = _correct_nutation_precession(
        v1950,
        orbit.nutation_precession
    )
    return _rotate_to_greenwich(
        vcorr,
        orbit.angles.greenwich_sidereal_time
    )


cdef Vector3D _get_satellite_z_axis_1950(
    double angle_between_sat_spin_and_z_axis,
    double angle_between_sat_spin_and_yz_plane
):
    """Get satellite z-axis (spin) in mean of 1950 coordinates."""
    alpha = angle_between_sat_spin_and_z_axis
    delta = angle_between_sat_spin_and_yz_plane
    cos_delta = cos(delta)
    return Vector3D(
        x=sin(delta),
        y=-cos_delta * sin(alpha),
        z=cos_delta * cos(alpha)
    )


cdef Vector3D _correct_nutation_precession(Vector3D vector, double[:, :] nutation_precession):
    return matrix_vector(nutation_precession, vector)


cdef Vector3D _rotate_to_greenwich(Vector3D vector, double greenwich_sidereal_time):
    cos_sid = cos(greenwich_sidereal_time)
    sin_sid = sin(greenwich_sidereal_time)
    rotated = Vector3D(
        x=cos_sid * vector.x + sin_sid * vector.y,
        y=-sin_sid * vector.x + cos_sid * vector.y,
        z=vector.z
    )
    return normalize_vector(rotated)

# TODO: Annotation
cdef Vector3D _get_satellite_unit_vector_x(Vector3D unit_vector_z, Attitude attitude, Orbit orbit):
    sat_sun_vec = _get_vector_from_satellite_to_sun(
        orbit.angles.declination_from_sat_to_sun,
        orbit.angles.right_ascension_from_sat_to_sun
    )
    return _get_unit_vector_x(
        sat_sun_vec,
        unit_vector_z,
        attitude.angle_between_earth_and_sun
    )


cdef Vector3D _get_vector_from_satellite_to_sun(
    double declination_from_sat_to_sun,
    double right_ascension_from_sat_to_sun
):
    declination = declination_from_sat_to_sun
    right_ascension = right_ascension_from_sat_to_sun
    cos_declination = cos(declination)
    return Vector3D(
        x=cos_declination * cos(right_ascension),
        y=cos_declination * sin(right_ascension),
        z=sin(declination)
    )


cdef Vector3D _get_unit_vector_x(
    Vector3D sat_sun_vec,
    Vector3D unit_vector_z,
    double angle_between_earth_and_sun

):
    beta = angle_between_earth_and_sun
    sin_beta = sin(beta)
    cos_beta = cos(beta)
    cross1 = _get_uz_cross_satsun(unit_vector_z, sat_sun_vec)
    cross2 = cross_product(cross1, unit_vector_z)
    unit_vector_x = Vector3D(
        x=sin_beta * cross1.x + cos_beta * cross2.x,
        y=sin_beta * cross1.y + cos_beta * cross2.y,
        z=sin_beta * cross1.z + cos_beta * cross2.z
    )
    return normalize_vector(unit_vector_x)


cdef Vector3D _get_uz_cross_satsun(Vector3D unit_vector_z, Vector3D sat_sun_vec):
    res = cross_product(unit_vector_z, sat_sun_vec)
    return normalize_vector(res)


cdef Vector3D _get_satellite_unit_vector_y(Vector3D unit_vector_x, Vector3D unit_vector_z):
    res = cross_product(unit_vector_z, unit_vector_x)
    return normalize_vector(res)


cdef Vector3D _get_earth_fixed_coords(Vector3D point, Vector3D unit_vector_x, Vector3D unit_vector_y, Vector3D unit_vector_z):
    ux, uy, uz = unit_vector_x, unit_vector_y, unit_vector_z
    # Multiply with matrix of satellite unit vectors [ux, uy, uz]
    return Vector3D(
        x=ux.x * point.x + uy.x * point.y + uz.x * point.z,
        y=ux.y * point.x + uy.y * point.y + uz.y * point.z,
        z=ux.z * point.x + uy.z * point.y + uz.z * point.z
    )


cdef Vector3D intersect_with_earth(Vector3D view_vector, Satpos sat_pos, EarthEllipsoid ellipsoid):
    """Intersect instrument viewing vector with the earth's surface.

    Reference: Appendix E, section 2.11 in the GMS user guide.

    Args:
        view_vector (Vector3D): Instrument viewing vector in earth-fixed
            coordinates.
        sat_pos (Vector3D): Satellite position in earth-fixed coordinates.
        ellipsoid (EarthEllipsoid): Earth ellipsoid.

    Returns:
        Intersection (Vector3D) with the earth's surface.
    """
    distance = _get_distance_to_intersection(view_vector, sat_pos, ellipsoid)
    return Vector3D(
        sat_pos.x + distance * view_vector.x,
        sat_pos.y + distance * view_vector.y,
        sat_pos.z + distance * view_vector.z
    )


cdef double _get_distance_to_intersection(Vector3D view_vector, Satpos sat_pos, EarthEllipsoid ellipsoid):
    """Get distance to intersection with the earth.

    If the instrument is pointing towards the earth, there will be two
    intersections with the surface. Choose the one on the instrument-facing
    side of the earth.
    """
    cdef double d1, d2
    d1, d2 = _get_distances_to_intersections(view_vector, sat_pos, ellipsoid)
    return min(d1, d2)

@cython.cdivision(True)
cdef _get_distances_to_intersections(Vector3D view_vector, Satpos sat_pos, EarthEllipsoid ellipsoid):
    """Get distances to intersections with the earth's surface.

    Returns:
        Distances to two intersections with the surface.
    """
    cdef double a, b, c
    a, b, c = _get_abc_helper(view_vector, sat_pos, ellipsoid)
    tmp = sqrt((b**2 - a * c))
    dist_1 = (-b + tmp) / a
    dist_2 = (-b - tmp) / a
    return dist_1, dist_2


cdef _get_abc_helper(Vector3D view_vector, Satpos sat_pos, EarthEllipsoid ellipsoid):
    """Get a,b,c helper variables.

    Reference: Appendix E, Equation (26) in the GMS user guide.
    """
    flat2 = (1 - ellipsoid.flattening) ** 2
    ux, uy, uz = view_vector.x, view_vector.y, view_vector.z
    x, y, z = sat_pos.x, sat_pos.y, sat_pos.z
    a = flat2 * (ux ** 2 + uy ** 2) + uz ** 2
    b = flat2 * (x * ux + y * uy) + z * uz
    c = flat2 * (x ** 2 + y ** 2 - ellipsoid.equatorial_radius ** 2) + z ** 2
    return a, b, c

@cython.cdivision(True)
cdef double rad2deg(double radians):
    """Convert radians to degrees."""
    return radians * (180.0 / M_PI)

cdef transform_earth_fixed_to_geodetic_coords(Vector3D point, double earth_flattening):
    """Transform from earth-fixed to geodetic coordinates.

    Args:
        point (Vector3D): Point in earth-fixed coordinates.
        earth_flattening: Flattening of the earth.

    Returns:
        Geodetic longitude and latitude (degrees).
    """
    x, y, z = point.x, point.y, point.z
    f = earth_flattening
    lon = atan2(y, x)
    lat = atan2(z, ((1 - f) ** 2 * sqrt(x**2 + y**2)))
    return rad2deg(lon), rad2deg(lat)


cdef Orbit interpolate_orbit_prediction(_OrbitPrediction orbit_prediction, double observation_time):
    """Interpolate orbit prediction at the given observation time."""
    angles = _interpolate_orbit_angles(observation_time, orbit_prediction)
    sat_position = _interpolate_sat_position(observation_time, orbit_prediction)
    nutation_precession = interpolate_nearest(
        observation_time,
        orbit_prediction.prediction_times,
        orbit_prediction.nutation_precession,
    )
    return Orbit(
        angles=angles,
        sat_position=sat_position,
        nutation_precession=nutation_precession,
    )


def _interpolate_orbit_angles(observation_time, orbit_prediction):
    sidereal_time = interpolate_angles(
        observation_time,
        orbit_prediction.prediction_times,
        orbit_prediction.angles.greenwich_sidereal_time,
    )
    declination = interpolate_angles(
        observation_time,
        orbit_prediction.prediction_times,
        orbit_prediction.angles.declination_from_sat_to_sun,
    )
    right_ascension = interpolate_angles(
        observation_time,
        orbit_prediction.prediction_times,
        orbit_prediction.angles.right_ascension_from_sat_to_sun,
    )
    return OrbitAngles(
        greenwich_sidereal_time=sidereal_time,
        declination_from_sat_to_sun=declination,
        right_ascension_from_sat_to_sun=right_ascension,
    )


def _interpolate_sat_position(observation_time, orbit_prediction):
    x = interpolate_continuous(
        observation_time,
        orbit_prediction.prediction_times,
        orbit_prediction.sat_position.x,
    )
    y = interpolate_continuous(
        observation_time,
        orbit_prediction.prediction_times,
        orbit_prediction.sat_position.y,
    )
    z = interpolate_continuous(
        observation_time,
        orbit_prediction.prediction_times,
        orbit_prediction.sat_position.z,
    )
    return Vector3D(x, y, z)


def interpolate_attitude_prediction(attitude_prediction, observation_time):
    """Interpolate attitude prediction at given observation time."""
    angle_between_earth_and_sun = interpolate_angles(
        observation_time,
        attitude_prediction.prediction_times,
        attitude_prediction.attitude.angle_between_earth_and_sun,
    )
    angle_between_sat_spin_and_z_axis = interpolate_angles(
        observation_time,
        attitude_prediction.prediction_times,
        attitude_prediction.attitude.angle_between_sat_spin_and_z_axis,
    )
    angle_between_sat_spin_and_yz_plane = interpolate_angles(
        observation_time,
        attitude_prediction.prediction_times,
        attitude_prediction.attitude.angle_between_sat_spin_and_yz_plane,
    )
    return Attitude(
        angle_between_earth_and_sun,
        angle_between_sat_spin_and_z_axis,
        angle_between_sat_spin_and_yz_plane,
    )


def interpolate_continuous(x, x_sample, y_sample):
    """Linear interpolation of continuous quantities.

    Numpy equivalent would be np.interp(..., left=np.nan, right=np.nan), but
    numba currently doesn't support those keyword arguments.
    """
    try:
        return _interpolate(x, x_sample, y_sample)
    except Exception:
        # Numba cannot distinguish exception types
        return np.nan

@cython.cdivision(True)
@cython.boundscheck(False)
cdef double _interpolate(double x, double[:] x_sample, double[:] y_sample):
    cdef int i = _find_enclosing_index(x, x_sample)
    offset = y_sample[i]
    x_diff = x_sample[i + 1] - x_sample[i]
    y_diff = y_sample[i + 1] - y_sample[i]
    slope = y_diff / x_diff
    dist = x - x_sample[i]
    return offset + slope * dist

@cython.boundscheck(False)
cdef int _find_enclosing_index(double x, double[:] x_sample):
    """Find where x_sample encloses x."""
    cdef int i
    cdef int n = x_sample.shape[0] - 1  # Use shape[0] for memoryview length
    for i in range(n):
        if x_sample[i] <= x < x_sample[i + 1]:
            return i
    raise ValueError("x not enclosed by x_sample")


def interpolate_angles(x, x_sample, y_sample):
    """Linear interpolation of angles.

    Requires 2-pi periodicity to be unwrapped before (for
    performance reasons). Interpolated angles are wrapped
    back to [-pi, pi] to restore periodicity.
    """
    return _wrap_2pi(interpolate_continuous(x, x_sample, y_sample))

@cython.cdivision(True)
@cython.boundscheck(False)
cdef double[:] _wrap_2pi(double[:] values):
    """Wrap values to interval [-pi, pi].

    Source: https://stackoverflow.com/a/15927914/5703449
    """
    cdef int n = values.shape[0] - 1
    cdef double[:] ret = np.zeros_like(values)
    for i in range(n):
        ret[i] = (values[i] + M_PI) % (2 * M_PI) - M_PI
    return ret


def interpolate_nearest(x, x_sample, y_sample):
    """Nearest neighbour interpolation."""
    try:
        return _interpolate_nearest(x, x_sample, y_sample)
    except Exception:
        return np.nan * np.ones_like(y_sample[0])


def _interpolate_nearest(x, x_sample, y_sample):
    i = _find_enclosing_index(x, x_sample)
    return y_sample[i]

@cython.boundscheck(False)
cdef Vector3D matrix_vector(double[:,:] m, Vector3D v):
    """Multiply (3,3)-matrix and Vector3D."""
    x = m[0, 0] * v.x + m[0, 1] * v.y + m[0, 2] * v.z
    y = m[1, 0] * v.x + m[1, 1] * v.y + m[1, 2] * v.z
    z = m[2, 0] * v.x + m[2, 1] * v.y + m[2, 2] * v.z
    return Vector3D(x, y, z)

@cython.cdivision(True)
cdef Vector3D cross_product(Vector3D a, Vector3D b):
    """Compute vector product a x b."""
    return Vector3D(
        x=a.y * b.z - a.z * b.y,
        y=a.z * b.x - a.x * b.z,
        z=a.x * b.y - a.y * b.x
    )

@cython.cdivision(True)
cdef Vector3D normalize_vector(Vector3D v):
    """Normalize a Vector3D."""
    cdef double norm = sqrt(v.x**2 + v.y**2 + v.z**2)
    return Vector3D(
        v.x / norm,
        v.y / norm,
        v.z / norm
    )

cdef class Projection:

    cdef readonly ImageOffset _image_offset
    cdef readonly EarthEllipsoid _earth_ellipsoid
    cdef readonly ScanningAngles _scanning_angles
    cdef readonly Attitude _attitude
    cdef readonly _AttitudePrediction _attitude_prediction
    cdef readonly OrbitAngles _orbit_angles
    cdef readonly Satpos _satpos
    cdef readonly _OrbitPrediction _orbit_prediction
    cdef readonly ProjectionParameters _proj_params
    cdef readonly ScanningParameters _scan_params
    cdef readonly StaticNavigationParameters _static_navigation_params
    cdef readonly PredictedNavigationParameters _predicted_navigation_params
    cdef readonly ImageNavigationParameters _navigation_parameters


    def __init__(self):
        pass

    def load_image_offset(self, double center_line_vissr_frame, double center_pixel_vissr_frame, double pixel_offset):
        self._image_offset = ImageOffset(
            line_offset=center_line_vissr_frame,
            pixel_offset=center_pixel_vissr_frame + pixel_offset,
        )

    def load_earth_ellipsoid(self):
        self._earth_ellipsoid = EarthEllipsoid(EARTH_FLATTENING, EARTH_EQUATORIAL_RADIUS)

    def load_scanning_angles(self, double stepping_angle, double sampling_angle, cnp.ndarray[double, ndim=2] misalignment):
        self._scanning_angles = ScanningAngles(stepping_angle, sampling_angle, misalignment)

    def load_attitude(self, cnp.ndarray[double, ndim=1] angle_between_earth_and_sun,
            cnp.ndarray[double, ndim=1] angle_between_sat_spin_and_z_axis,
            cnp.ndarray[double, ndim=1] angle_between_sat_spin_and_yz_plane):
        self._attitude = Attitude(angle_between_earth_and_sun, angle_between_sat_spin_and_z_axis, angle_between_sat_spin_and_yz_plane)

    def load_attitude_prediction(self, double prediction_times):
        self._attitude_prediction = AttitudePrediction(prediction_times, self._attitude).to_numba()

    def load_orbit_angles(self, cnp.ndarray[double, ndim=1] greenwich_sidereal_time,
            cnp.ndarray[double, ndim=1] declination_from_sat_to_sun,
            cnp.ndarray[double, ndim=1] right_ascension_from_sat_to_sun):
        self._orbit_angles = OrbitAngles(greenwich_sidereal_time, declination_from_sat_to_sun, right_ascension_from_sat_to_sun)

    def load_satpos(self, cnp.ndarray[double, ndim=1] x, cnp.ndarray[double, ndim=1] y, cnp.ndarray[double, ndim=1] z):
        self._satpos = Satpos(x, y, z)

    def load_orbit_prediction(self, double prediction_times, cnp.ndarray[double, ndim=2] nutation_precession):
        self._orbit_prediction = OrbitPrediction(prediction_times, self._orbit_angles, self._satpos, nutation_precession).to_numba()

    def load_proj_params(self):
        self._proj_params = ProjectionParameters(
            image_offset=self._image_offset,
            scanning_angles=self._scanning_angles,
            earth_ellipsoid=self._earth_ellipsoid
        )
    
    def load_scanning_params(self, double start_time_of_scan, double spinning_rate, int num_sensors, double sampling_angle):
        self._scan_params = ScanningParameters(start_time_of_scan, spinning_rate, num_sensors, sampling_angle)

    def load_static_navigation_params(self):
        self._static_navigation_params = StaticNavigationParameters(self._proj_params, self._scan_params)

    def load_predicted_navigation_params(self):
        self._predicted_navigation_params = PredictedNavigationParameters(
            attitude=attitude_prediction,
            orbit=orbit_prediction
        )
    def load_navigation_parameters(self):
        self._navigation_parameters = ImageNavigationParameters(self._static_navigation_params, self._predicted_navigation_params)

    def get_lon_lat(self, lines, pixels):
        return get_lons_lats(lines, pixels, self._navigation_parameters)