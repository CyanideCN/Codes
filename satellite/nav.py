import numpy as np

MCMISSING = int("0x80808080", 16)

def icon1(yymmdd):
    # Define the number of days cumulatively at the start of each month
    num = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]

    # Extract year, month, and day from input
    year = (yymmdd // 10000) % 100
    month = (yymmdd // 100) % 100
    day = yymmdd % 100

    # Validate month
    if month < 1 or month > 12:
        month = 1

    # Calculate Julian day
    julday = day + num[month - 1]  # Adjust index for 0-based Python lists

    # Check for leap year and adjust if necessary
    if year % 4 == 0 and month > 2:
        julday += 1

    # Combine year with Julian day
    icon1 = 1000 * year + julday
    return icon1

def leapyr(iy):
    """Determine if a year is a leap year."""
    return 366 - (iy % 4 + 3) // 4

def itime(time):
    """Convert time to integer format HHMMSS."""
    hours = int(time)
    minutes = int((time - hours) * 60)
    seconds = int(((time - hours) * 60 - minutes) * 60)
    return hours * 10000 + minutes * 100 + seconds

def flalo(m):
    """
    Convert an integer angle (format DDDMMSS) or time (format HHMMSS) to float.

    Parameters:
    - m (int): Angle or time as an integer.

    Returns:
    - float: The angle or time converted to decimal degrees or hours.
    """
    n = abs(m)
    # Convert integer to degrees/hours, minutes, seconds and then to float
    flalo_value = (n // 10000) + ((n // 100) % 100) / 60.0 + (n % 100) / 3600.0
    # Apply the sign of the original input
    if m < 0:
        flalo_value = -flalo_value
    return flalo_value

def epoch(ietimy, ietimh, semima, oeccen, xmeana):
    """Find time of perigee from Keplerian epoch."""
    pi = 3.14159265
    rdpdg = pi / 180.0
    re = 6378.388
    gracon = 0.07436574

    xmmc = gracon * np.sqrt(re / semima)**3
    xmanom = rdpdg * xmeana
    time = (xmanom - oeccen * np.sin(xmanom)) / (60.0 * xmmc)
    time1 = flalo(ietimh)
    time = time1 - time
    iday = 0

    if time > 48.0:
        time -= 48.0
        iday = 2
    elif time > 24.0:
        time -= 24.0
        iday = 1
    elif time < -24.0:
        time += 48.0
        iday = -2
    elif time < 0.0:
        time += 24.0
        iday = -1

    ietimh = itime(time)

    if iday == 0:
        return ietimy, ietimh

    jyear = (ietimy // 1000) % 100
    jday = ietimy % 1000
    jday += iday

    if jday < 1:
        jyear -= 1
        jday += leapyr(jyear)
    else:
        jtot = leapyr(jyear)
        if jday > jtot:
            jyear += 1
            jday -= jtot

    ietimy = 1000 * jyear + jday
    return ietimy, ietimh

def timdif(iyrda1, ihms1, iyrda2, ihms2):
    iy1 = (iyrda1 // 1000) % 100
    id1 = iyrda1 % 1000
    ifac1 = (iy1 - 1) // 4 + 1
    d1 = 365 * (iy1 - 1) + ifac1 + id1 - 1
    
    iy2 = (iyrda2 // 1000) % 100
    id2 = iyrda2 % 1000
    ifac2 = (iy2 - 1) // 4 + 1
    d2 = 365 * (iy2 - 1) + ifac2 + id2 - 1
    
    t1 = 1440.0 * d1 + 60.0 * flalo(ihms1)
    t2 = 1440.0 * d2 + 60.0 * flalo(ihms2)
    
    timdif = t2 - t1
    return timdif

def racrae(iyrdy, ihms, rac):
    """Converts Celestial Longitude to Earth Longitude"""
    # Constants
    sha = 100.26467  # Stellar Hour Angle
    irayd = 74001  # Reference year and day
    irahms = 0  # Reference time
    solsid = 1.00273791  # Solar sidereal time ratio

    # Calculate the Right Ascension Hour Angle (RAHA)
    # Using the previously defined TIMDIF function to calculate time difference in minutes
    raha = rac - sha + timdif(iyrdy, ihms, irayd, irahms) * solsid / 4.0

    # Ensure the Earth Longitude (RAE) is within 0 to 360 degrees
    rae = raha % 360.0
    if rae < 0.0:
        rae += 360.0

    return rae

def raerac(iyrdy, ihms, rae):
    """Converts Earth Longitude to Celestial Longitude"""
    sha = 100.26467
    irayd = 74001
    irahms = 0
    solsid = 1.00273791
    raha = rae + timdif(irayd, irahms, iyrdy, ihms) * solsid / 4.0 + sha
    rac = raha % 360.0
    if rac < 0.0:
        rac += 360.0
    return rac

def nxyzll(x, y, z):
    # Constants
    rdpdg = 1.745329252e-2
    asq = 40683833.48
    bsq = 40410330.18
    ab = 40546851.22

    if x == 0 and y == 0 and z == 0:
        return 100.0, 200.0  # Default values as per the Fortran code

    # Convert to geodetic latitude
    a = np.arctan(z / np.sqrt(x**2 + y**2))
    xlat = np.arctan2(asq * np.sin(a), bsq * np.cos(a)) / rdpdg

    # Convert to longitude
    xlon = -np.arctan2(y, x) / rdpdg

    # Adjusting xlon to ensure it's in the range of -180 to 180 degrees
    if xlon < -180:
        xlon += 360
    elif xlon > 180:
        xlon -= 360

    # Ensure North and West are positive as per comment in Fortran code
    # Note: In the typical geographical coordinate system, East and North are positive.
    # This adjustment assumes West as positive contrary to the standard, which might be an oversight or specific use case in the original code.
    # Here we assume the standard convention where East (positive longitude) and North (positive latitude) are positive.

    return xlat, xlon

class Navigation:

    isLineFlipped = False
    lineOffset = 0.0
    resLine = 1.0
    resElement = 1
    magLine = 1.0
    magElement = 1.0
    startLine = 0
    startElement = 0.0
    startImageLine = 0.0
    startImageElement = 0.0

    def __init__(self, navblock):
        jday = navblock[1]
        jtime = navblock[2]

        # INTIALIZE NAVCOM
        self.navday = jday % 100000
        ietimy = icon1(navblock[4])
        ietimh = 100 * (navblock[5] / 100) + round(0.6 * (navblock[5] % 100))
        self.semima = navblock[6] / 100.0
        self.oeccen = navblock[7] / 1000000.0
        self.orbinc = navblock[8] / 1000.0
        self.xmeana = navblock[9] / 1000.0
        self.perhel = navblock[10] / 1000.0
        self.asnode = navblock[11] / 1000.0
        self.ietimy, self.ietimh = epoch(ietimy, ietimh, self.semima, self.oeccen, self.xmeana)
        self.declin = flalo(navblock[12])
        self.rascen = flalo(navblock[13])
        self.piclin = navblock[14]
        if navblock[14] > 1000000:
            self.piclin /= 10000
        if navblock[12] == 0 and navblock[13] == 0 and navblock[14] == 0:
            raise ValueError("Invalid ascension/declination parameters")
        if navblock[15] == 0:
            raise ValueError("Invalid spin period")
        self.spinra = navblock[15] / 1000.0
        if navblock[15] != 0 and self.spinra < 300:
            self.spinra = 60000.0 / self.spinra
        self.deglin = flalo(navblock[16])
        self.lintot = navblock[17]
        self.degele = flalo(navblock[18])
        self.ieltot = navblock[19]
        self.pitch = flalo(navblock[20])
        self.yaw = flalo(navblock[21])
        self.roll = flalo(navblock[22])
        self.skew = navblock[28] / 100000.0
        if navblock[28] == MCMISSING:
            self.skew = 0.0
        # BETCOM
        self.iajust = navblock[24]
        self.iseang = navblock[27]
        self.ibtcon = 6289920
        self.negbet = 3144960
        # NAVINI
        self.emega = 0.26251617
        self.ab = 40546851.22
        self.asq = 40683833.48
        self.bsq = 40410330.18
        self.r = 6371.22
        self.rsq = self.r ** 2
        self.rdpdg = 1.745329252e-02
        self.numsen = (self.lintot / 100000) % 100
        if self.numsen < 1:
            self.numsen = 1
        self.totlin = self.numsen * (self.lintot % 100000)
        self.radlin = self.rdpdg * self.deglin / (self.totlin - 1.0)
        self.totele = self.ieltot
        self.radele = self.rdpdg * self.degele / (self.totele - 1.0)
        self.picele = (1.0 + self.totele) / 2.0
        self.cpitch = self.rdpdg * self.pitch
        self.cyaw = self.rdpdg * self.yaw
        self.croll = self.rdpdg * self.roll
        self.pskew = np.arctan2(self.skew, self.radlin / self.radele)
        stp = np.sin(self.cpitch)
        ctp = np.cos(self.cpitch)
        sty = np.sin(self.cyaw - self.pskew)
        cty = np.cos(self.cyaw - self.pskew)
        _str = np.sin(self.croll)
        ctr = np.cos(self.croll)
        self.rotm11 = ctr * ctp
        self.rotm13 = sty * _str * ctp + cty * stp
        self.rotm21 = -_str
        self.rotm23 = sty * ctr
        self.rotm31 = -ctr * stp
        self.rotm33 = cty * ctp - sty * _str * stp
        self.rfact = self.rotm31 ** 2 + self.rotm33 ** 2
        self.roasin = np.arctan2(self.rotm31, self.rotm33)
        self.tmpscl = self.spinra / 3600000.0
        dec = self.declin * self.rdpdg
        sindec = np.sin(dec)
        cosdec = np.cos(dec)
        ras = self.rascen * self.rdpdg
        sinras = np.sin(ras)
        cosras = np.cos(ras)
        self.b11 = -sinras
        self.b12 = cosras
        self.b13 = 0.0
        self.b21 = -sindec * cosras
        self.b22 = -sindec * sinras
        self.b23 = cosdec
        self.b31 = cosdec * cosras
        self.b32 = cosdec * sinras
        self.b33 = sindec
        self.xref = raerac(self.navday, 0, 0) * self.rdpdg
        # TIME SPECIFIC
        self.pictim = flalo(jtime)
        self.gamma = navblock[38] / 100.0
        self.gamdot = navblock[39] / 100.0
        # VASCOM
        iss = jday / 100000
        if (iss > 25 or iss == 12) and navblock[30] > 0:
            #       THIS SECTION DOES VAS BIRDS AND GMS
            #       IT USES TIMES AND SCAN LINE FROM BETA RECORDS
            self.scan1 = float(navblock[30])
            self.time1 = flalo(navblock[31])
            self.scan2 = float(navblock[34])
            self.time2 = flalo(navblock[35])
        else:
            # THIS SECTION DOES THE OLD GOES BIRDS
            self.scan1 = 1.0
            self.time1 - flalo(jtime)
            self.scan2 = float(self.lintot % 100000)
            self.time2 = self.time1 + self.scan2 * self.tmpscl

        self.iold = 0

    def satvec(self, samtim):
        # Constants
        pi = np.pi
        twopi = 2.0 * pi
        pi720 = pi / 720.0
        rdpdg = pi / 180.0
        re = 6378.388
        gracon = 0.07436574
        solsid = 1.00273791
        sha = 100.26467 * rdpdg
        irayd = 74001
        irahms = 0
        o = rdpdg * self.orbinc
        p = rdpdg * self.perhel
        a = rdpdg * self.asnode
        so = np.sin(o)
        co = np.cos(o)
        sp = np.sin(p) * self.semima
        cp = np.cos(p) * self.semima
        sa = np.sin(a)
        ca = np.cos(a)
        px = cp * ca - sp * sa * co
        py = cp * sa + sp * ca * co
        pz = sp * so
        qx = -sp * ca - cp * sa * co
        qy = -sp * sa + cp * ca * co
        qz = cp * so
        srome2 = np.sqrt(1.0 - self.oeccen) * np.sqrt(1.0 + self.oeccen)
        xmmc = gracon * re * np.sqrt(re / self.semima) / self.semima
        iey = (self.ietimy // 1000) % 100
        ied = self.ietimy % 1000
        iefac = (iey - 1) // 4 + 1
        de = 365 * (iey - 1) + iefac + ied - 1
        # te = 1440.0 * de + 60.0 * (navcom['IETIMH'] // 100)  # Assuming FLALO is a conversion, simplified here
        te = 1440.0 * de + 60.0 * flalo(self.ietimh)
        iray = irayd // 1000
        irad = irayd % 1000
        irafac = (iray - 1) // 4 + 1
        dra = 365 * (iray - 1) + irafac + irad - 1
        # tra = 1440.0 * dra + 60.0 * (irahms // 100)  # Assuming FLALO conversion, simplified
        tra = 1440.0 * dra + 60.0 * flalo(irahms)
        inavy = (self.navday // 1000) % 100
        inavd = self.navday % 1000
        infac = (inavy - 1) // 4 + 1
        dnav = 365 * (inavy - 1) + infac + inavd - 1
        tdife = dnav * 1440. - te
        tdifra = dnav * 1440. - tra
        epsiln = 1.0e-8
        timsam = samtim * 60.0
        diftim = tdife + timsam
        xmanom = xmmc * diftim
        ecanm1 = xmanom

        # Iterative solution for the Eccentric Anomaly
        for _ in range(20):
            ecanom = xmanom + self.oeccen * np.sin(ecanm1)
            if np.abs(ecanom - ecanm1) < epsiln:
                break
            ecanm1 = ecanom

        xomega = np.cos(ecanom) - self.oeccen
        yomega = srome2 * np.sin(ecanom)
        z = xomega * pz + yomega * qz
        y = xomega * py + yomega * qy
        x = xomega * px + yomega * qx

        return x, y, z
    
    def nvxsae(self, xlin, xele):
        # Constants
        pi = np.pi

        # Convert xlin to nearest integer and calculate related parameters
        ilin = int(round(xlin))
        
        parlin = (ilin - 1) / self.numsen + 1
        framet = self.tmpscl * parlin
        samtim = framet + self.pictim

        # Call SATVEC to get satellite coordinates
        xsat, ysat, zsat = self.satvec(samtim)  # Assuming SATVEC returns a tuple (X, Y, Z)

        # Calculations for line and element
        ylin = (xlin - self.piclin) * self.radlin
        yele = (xele - self.picele + self.gamma + self.gamdot * samtim) * self.radele

        # Coordinate transformations
        xcor = self.b11 * xsat + self.b12 * ysat + self.b13 * zsat
        ycor = self.b21 * xsat + self.b22 * ysat + self.b23 * zsat
        rot = np.arctan2(ycor, xcor) + pi
        yele = yele - rot

        # More transformations
        coslin = np.cos(ylin)
        sinlin = np.sin(ylin)
        sinele = np.sin(yele)
        cosele = np.cos(yele)

        eli = self.rotm11 * coslin - self.rotm13 * sinlin
        emi = self.rotm21 * coslin - self.rotm23 * sinlin
        eni = self.rotm31 * coslin - self.rotm33 * sinlin

        temp = eli
        eli = cosele * eli + sinele * emi
        emi = -sinele * temp + cosele * emi

        elo = self.b11 * eli + self.b21 * emi + self.b31 * eni
        emo = self.b12 * eli + self.b22 * emi + self.b32 * eni
        eno = self.b13 * eli + self.b23 * emi + self.b33 * eni

        basq = self.bsq / self.asq
        onemsq = 1.0 - basq
        aq = basq + onemsq * eno**2
        bq = 2.0 * ((elo * xsat + emo * ysat) * basq + eno * zsat)
        cq = (xsat**2 + ysat**2) * basq + zsat**2 - self.bsq
        rad = bq**2 - 4.0 * aq * cq

        if rad < 1.0:
            return np.nan, np.nan  # Indicating an error or "off of Earth"

        s = -(bq + np.sqrt(rad)) / (2.0 * aq)
        x = xsat + elo * s
        y = ysat + emo * s
        z = zsat + eno * s

        ct = np.cos(self.emega * samtim + self.xref)
        st = np.sin(self.emega * samtim + self.xref)

        x1 = ct * x + st * y
        y1 = -st * x + ct * y

        #if nvunit['LLSW'] == 0:
        xpar, ypar = nxyzll(x1, y1, z)  # Assuming NXYZLL returns latitude and longitude
        # zpar = 0.0
        # else:
        #    xpar, ypar, zpar = x1, y1, z

        # return 0, xpar, ypar, zpar  # Function is successful
        return xpar, ypar
    
    def proj(self, lines, elements):
        out_shape = lines.shape
        out_size = lines.size
        out_x = np.ones(out_size)
        out_y = np.ones(out_size)
        ll = lines.ravel()
        el = elements.ravel()
        # TODO: change it back when finished
        ie, il = self.area_coord_to_img_coord(ll, el)
        for index in range(out_size):
            out_x[index], out_y[index] = self.nvxsae(il[index], ie[index])
        return out_x.reshape(out_shape), out_y.reshape(out_shape)
    
    def area_coord_to_img_coord(self, line, element):
        # Handle flipped coordinates for lines
        if self.isLineFlipped:
            line = self.lineOffset - line

        # Compute new values for line and element coordinates
        new_line_coords = self.startImageLine + (self.resLine * (line - self.startLine)) / self.magLine
        new_element_coords = self.startImageElement + (self.resElement * (element - self.startElement)) / self.magElement

        # Use np.isnan to filter out NaN values, then apply computations
        valid_line_indices = ~np.isnan(line)
        valid_element_indices = ~np.isnan(element)

        # Initialize arrays with NaNs
        new_line_vals = np.full_like(line, np.nan)
        new_element_vals = np.full_like(element, np.nan)

        # Update only the valid indices
        new_line_vals[valid_line_indices] = new_line_coords[valid_line_indices]
        new_element_vals[valid_element_indices] = new_element_coords[valid_element_indices]
        # pyarea and mcidas use element first
        # TODO: change it back when finished
        return new_element_vals, new_line_vals

    def set_image_start(self, start_line, start_elem):
        self.startImageLine = float(start_line)
        self.startImageElement = float(start_elem)