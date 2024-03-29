C
C THIS ROUTINE SETS UP COMMON BLOCKS NAVCOM AND NAVINI FOR USE BY THE
C NAVIGATION TRANSFORMATION ROUTINES NVXSAE AND NVXEAS.
C NVXINI SHOULD BE RECALLED EVERY TIME A TRANSFORMATION IS DESIRED
C FOR A PICTURE WITH A DIFFERENT TIME THAN THE PREVIOUS CALL.
C IFUNC IS 1 (INITIALIZE; SET UP COMMON BLOCKS)
C          2 (ACCEPT/PRODUCE ALL EARTH COORDINATES IN LAT/LON
C            FORM IF IARR IS 'LL  ' OR IN THE X,Y,Z COORDINATE FRAME
C            IF IARR IS 'XYZ '.
C            THIS AFFECTS ALL SUBSEQUENT NVXEAS OR NVXSAE CALLS.)
C IARR IS AN INTEGER ARRAY (DIM 128) IF IFUNC=1, CONTAINING NAV
C        PARAMETERS
C
      INTEGER IARR(*),GOES,LL,XYZ
      CHARACTER*2 CLLSW
      COMMON/NAVCOM/NAVDAY,LINTOT,DEGLIN,IELTOT,DEGELE,SPINRA,IETIMY,IET
     1IMH,SEMIMA,OECCEN,ORBINC,PERHEL,ASNODE,NOPCLN,DECLIN,RASCEN,PICLIN
     2,PRERAT,PREDIR,PITCH,YAW,ROLL,SKEW
      COMMON /BETCOM/IAJUST,IBTCON,NEGBET,ISEANG
      COMMON /VASCOM/SCAN1,TIME1,SCAN2,TIME2
      COMMON /NAVINI/
     1  EMEGA,AB,ASQ,BSQ,R,RSQ,
     2  RDPDG,
     3  NUMSEN,TOTLIN,RADLIN,
     4  TOTELE,RADELE,PICELE,
     5  CPITCH,CYAW,CROLL,
     6  PSKEW,
     7  RFACT,ROASIN,TMPSCL,
     8  B11,B12,B13,B21,B22,B23,B31,B32,B33,
     9  GAMMA,GAMDOT,
     A  ROTM11,ROTM13,ROTM21,ROTM23,ROTM31,ROTM33,
     B  PICTIM,XREF
      COMMON /NVUNIT/ LLSW
      DATA MISVAL/Z80808080/
      DATA JINIT/0/,GOES/4HGOES/,LL/4HLL  /,XYZ/4HXYZ /
C
      IF (JINIT.EQ.0) THEN
         JINIT=1
         LLSW=0
         JDAYSV=-1
         JTIMSV=-1
      ENDIF
      IF (IFUNC.EQ.2) THEN
         IF (IARR(1).EQ.LL) LLSW=0
         IF (IARR(1).EQ.XYZ) LLSW=1
         NVXINI=0
         RETURN
      ENDIF
      JTYPE=IARR(1)
      IF (JTYPE.NE.GOES) GOTO 90
      JDAY=IARR(2)
      JTIME=IARR(3)
      IF(JDAY.EQ.JDAYSV.AND.JTIME.EQ.JTIMSV) GO TO 10
C
C-----INITIALIZE NAVCOM
      NAVDAY=MOD(JDAY,100000)
      DO 20 N=7,12
      IF(IARR(N).GT.0) GO TO 25
   20 CONTINUE
      GO TO 90
   25 IETIMY=ICON1(IARR(5))
      IETIMH=100*(IARR(6)/100)+NINT(.6*MOD(IARR(6),100))
      SEMIMA=FLOAT(IARR(7))/100.0
      OECCEN=FLOAT(IARR(8))/1000000.0
      ORBINC=FLOAT(IARR(9))/1000.0
      XMEANA=FLOAT(IARR(10))/1000.0
      PERHEL=FLOAT(IARR(11))/1000.0
      ASNODE=FLOAT(IARR(12))/1000.0
      CALL EPOCH(IETIMY,IETIMH,SEMIMA,OECCEN,XMEANA)
      IF (IARR(5).EQ.0.OR.IARR(9).EQ.0) GOTO 90
      DECLIN=FLALO(IARR(13))
      RASCEN=FLALO(IARR(14))
      PICLIN=IARR(15)
      IF (IARR(15).GE.1000000) PICLIN=PICLIN/10000.
      IF (IARR(13).EQ.0.AND.IARR(14).EQ.0.AND.IARR(15).EQ.0)
     *   GOTO 90
      SPINRA=IARR(16)/1000.0
      IF(IARR(16).NE.0.AND.SPINRA.LT.300.0) SPINRA=60000.0/SPINRA
      IF (IARR(16).EQ.0) GOTO 90
      DEGLIN=FLALO(IARR(17))
      LINTOT=IARR(18)
      DEGELE=FLALO(IARR(19))
      IELTOT=IARR(20)
      PITCH=FLALO(IARR(21))
      YAW=FLALO(IARR(22))
      ROLL=FLALO(IARR(23))
      SKEW=IARR(29)/100000.0
      IF (IARR(29).EQ.MISVAL) SKEW=0.
C
C-----INITIALIZE BETCOM
      IAJUST=IARR(25)
      ISEANG=IARR(28)
      IBTCON=6289920
      NEGBET=3144960
C
C-----INITIALIZE NAVINI COMMON BLOCK
      EMEGA=.26251617
      AB=40546851.22
      ASQ=40683833.48
      BSQ=40410330.18
      R=6371.221
      RSQ=R*R
      RDPDG=1.745329252E-02
      NUMSEN=MOD(LINTOT/100000,100)
      IF(NUMSEN.LT.1)NUMSEN=1
      TOTLIN=NUMSEN*MOD(LINTOT,100000)
      RADLIN=RDPDG*DEGLIN/(TOTLIN-1.0)
      TOTELE=IELTOT
      RADELE=RDPDG*DEGELE/(TOTELE-1.0)
      PICELE=(1.0+TOTELE)/2.0
      CPITCH=RDPDG*PITCH
      CYAW=RDPDG*YAW
      CROLL=RDPDG*ROLL
      PSKEW=ATAN2(SKEW,RADLIN/RADELE)
      STP=SIN(CPITCH)
      CTP=COS(CPITCH)
      STY=SIN(CYAW-PSKEW)
      CTY =COS(CYAW-PSKEW)
      STR=SIN(CROLL)
      CTR=COS(CROLL)
      ROTM11=CTR*CTP
      ROTM13=STY*STR*CTP+CTY*STP
      ROTM21=-STR
      ROTM23=STY*CTR
      ROTM31=-CTR*STP
      ROTM33=CTY*CTP-STY*STR*STP
      RFACT=ROTM31**2+ROTM33**2
      ROASIN=ATAN2(ROTM31,ROTM33)
      TMPSCL=SPINRA/3600000.0
      DEC=DECLIN*RDPDG
      SINDEC=SIN(DEC)
      COSDEC=COS(DEC)
      RAS=RASCEN*RDPDG
      SINRAS=SIN(RAS)
      COSRAS=COS(RAS)
      B11=-SINRAS
      B12=COSRAS
      B13=0.0
      B21=-SINDEC*COSRAS
      B22=-SINDEC*SINRAS
      B23=COSDEC
      B31=COSDEC*COSRAS
      B32=COSDEC*SINRAS
      B33=SINDEC
      XREF=RAERAC(NAVDAY,0,0.0)*RDPDG
C
C-----TIME-SPECIFIC PARAMETERS (INCL GAMMA)
      PICTIM=FLALO(JTIME)
      GAMMA=FLOAT(IARR(39))/100.
      GAMDOT=FLOAT(IARR(40))/100.
C
C-----INITIALIZE VASCOM
      IF (JDAY/100000.GT.25.AND.IARR(31).GT.0) THEN
C        THIS SECTION DOES VAS BIRDS
C        IT USES TIMES AND SCAN LINE FROM BETA RECORDS
         SCAN1=FLOAT(IARR(31))
         TIME1=FLALO(IARR(32))
         SCAN2=FLOAT(IARR(35))
         TIME2=FLALO(IARR(36))
      ELSE
C        THIS SECTION DOES THE OLD GOES BIRDS
         SCAN1=1.
         TIME1=FLALO(JTIME)
         SCAN2=FLOAT(MOD(LINTOT,100000))
         TIME2=TIME1+SCAN2*TMPSCL
      ENDIF
C
C-----ALL DONE. EVERYTHING OK
 10   CONTINUE
      JDAYSV=JDAY
      JTIMSV=JTIME
      NVXINI=0
      RETURN
 90   NVXINI=-1
      RETURN
      END
C ********************************************************************
C
      FUNCTION NVXSAE(XLIN,XELE,XDUM,XPAR,YPAR,ZPAR)
C TRANSFORMS SAT COOR TO EARTH COOR.
C ALL PARAMETERS REAL*4
C INPUTS:
C XLIN,XELE ARE SATELLITE LINE AND ELEMENT (IMAGE COORDS.)
C XDUM IS DUMMY (IGNORE)
C OUTPUTS:
C XPAR,YPAR,ZPAR REPRESENT EITHER LAT,LON,(DUMMY) OR X,Y,Z DEPENDING
C ON THE OPTION SET IN PRIOR NVXINI CALL WITH IFUNC=2.
C FUNC VAL IS 0 (OK) OR -1 (CAN'T; E.G. OFF OF EARTH)
C
      COMMON/NAVCOM/NAVDAY,LINTOT,DEGLIN,IELTOT,DEGELE,SPINRA,IETIMY,IET
     1IMH,SEMIMA,OECCEN,ORBINC,PERHEL,ASNODE,NOPCLN,DECLIN,RASCEN,PICLIN
     2,PRERAT,PREDIR,PITCH,YAW,ROLL,SKEW
      COMMON/NAVINI/
     1  EMEGA,AB,ASQ,BSQ,R,RSQ,
     2  RDPDG,
     3  NUMSEN,TOTLIN,RADLIN,
     4  TOTELE,RADELE,PICELE,
     5  CPITCH,CYAW,CROLL,
     6  PSKEW,
     7  RFACT,ROASIN,TMPSCL,
     8  B11,B12,B13,B21,B22,B23,B31,B32,B33,
     9  GAMMA,GAMDOT,
     A  ROTM11,ROTM13,ROTM21,ROTM23,ROTM31,ROTM33,
     B  PICTIM,XREF
      COMMON /NVUNIT/ LLSW
      DATA PI/3.14159265/
C
      ILIN=NINT(XLIN)
      PARLIN=(ILIN-1)/NUMSEN+1
      FRAMET=TMPSCL*PARLIN
      SAMTIM=FRAMET+PICTIM
      CALL SATVEC(SAMTIM,XSAT,YSAT,ZSAT)
      YLIN=(XLIN-PICLIN)*RADLIN
      YELE=(XELE-PICELE+GAMMA+GAMDOT*SAMTIM)*RADELE
      XCOR=B11*XSAT+B12*YSAT+B13*ZSAT
      YCOR=B21*XSAT+B22*YSAT+B23*ZSAT
      ROT=ATAN2(YCOR,XCOR)+PI
      YELE=YELE-ROT
      COSLIN=COS(YLIN )
      SINLIN=SIN(YLIN)
      SINELE=SIN(YELE)
      COSELE=COS(YELE)
      ELI=ROTM11*COSLIN-ROTM13*SINLIN
      EMI=ROTM21*COSLIN-ROTM23*SINLIN
      ENI=ROTM31*COSLIN-ROTM33*SINLIN
      TEMP=ELI
      ELI=COSELE*ELI+SINELE*EMI
      EMI=-SINELE*TEMP+COSELE*EMI
      ELO=B11*ELI+B21*EMI+B31*ENI
      EMO=B12*ELI+B22*EMI+B32*ENI
      ENO=B13*ELI+B23*EMI+B33*ENI
      BASQ=BSQ/ASQ
      ONEMSQ=1.0-BASQ
      AQ=BASQ+ONEMSQ*ENO**2
      BQ=2.0*((ELO*XSAT+EMO*YSAT)*BASQ+ENO*ZSAT)
      CQ=(XSAT**2+YSAT**2)*BASQ+ZSAT**2-BSQ
      RAD=BQ**2-4.0*AQ*CQ
      IF(RAD.LT.1.0)GO TO 2
      S=-(BQ+SQRT(RAD))/(2.0*AQ)
      X=XSAT+ELO*S
      Y=YSAT+EMO*S
      Z=ZSAT+ENO*S
      CT=COS(EMEGA*SAMTIM+XREF)
      ST=SIN(EMEGA*SAMTIM+XREF)
      X1=CT*X+ST*Y
      Y1=-ST*X+CT*Y
      IF (LLSW.EQ.0) THEN
         CALL NXYZLL(X1,Y1,Z,XPAR,YPAR)
         ZPAR=0.
      ELSE
         XPAR=X1
         YPAR=Y1
         ZPAR=Z
      ENDIF
      NVXSAE=0
      RETURN
    2 NVXSAE=-1
      RETURN
      END
C ********************************************************************
C
      FUNCTION NVXEAS(XPAR,YPAR,ZPAR,XLIN,XELE,XDUM)
C 5/26/82;  TRANSFORM EARTH TO SATELLITE COORDS
C ALL PARAMETERS REAL*4
C INPUTS:
C XPAR,YPAR,ZPAR REPRESENT EITHER LAT,LON,(DUMMY) OR X,Y,Z DEPENDING
C ON THE OPTION SET IN PRIOR NVXINI CALL WITH IFUNC=2.
C OUTPUTS:
C XLIN,XELE ARE SATELLITE LINE AND ELEMENT (IMAGE COORDS.)
C XDUM IS DUMMY (IGNORE)
C FUNC VAL IS 0 (OK) OR -1 (CAN'T; E.G. BAD LAT/LON)
C
      COMMON/NAVINI/
     1  EMEGA,AB,ASQ,BSQ,R,RSQ,
     2  RDPDG,
     3  NUMSEN,TOTLIN,RADLIN,
     4  TOTELE,RADELE,PICELE,
     5  CPITCH,CYAW,CROLL,
     6  PSKEW,
     7  RFACT,ROASIN,TMPSCL,
     8  B11,B12,B13,B21,B22,B23,B31,B32,B33,
     9  GAMMA,GAMDOT,
     A  ROTM11,ROTM13,ROTM21,ROTM23,ROTM31,ROTM33,
     B  PICTIM,XREF
      COMMON/NAVCOM/NAVDAY,LINTOT,DEGLIN,IELTOT,DEGELE,SPINRA,IETIMY,IET
     1IMH,SEMIMA,OECCEN,ORBINC,PERHEL,ASNODE,NOPCLN,DECLIN,RASCEN,PICLIN
     2,PRERAT,PREDIR,PITCH,YAW,ROLL,SKEW
      COMMON/VASCOM/SCAN1,TIME1,SCAN2,TIME2
      COMMON /NVUNIT/ LLSW
      DATA OLDLIN/910./,ORBTIM/-99999./,MAXITR/5/
C
      NVXEAS=0
      IF (LLSW.EQ.0) THEN
         IF (ABS(XPAR).GT.90.) THEN
            NVXEAS=-1
            RETURN
         ENDIF
         CALL NLLXYZ(XPAR,YPAR,X1,Y1,Z)
      ELSE
         X1=XPAR
         Y1=YPAR
         Z=ZPAR
      ENDIF
      XDUM=0.0
      SAMTIM=TIME1
      DO 50 I=1,2
      IF(ABS(SAMTIM-ORBTIM).LT.0.0005) GO TO 10
      CALL SATVEC(SAMTIM,XSAT,YSAT,ZSAT)
      ORBTIM=SAMTIM
      XHT=SQRT(XSAT**2+YSAT**2+ZSAT**2)
   10 CT=COS(EMEGA*SAMTIM+XREF)
      ST=SIN(EMEGA*SAMTIM+XREF)
      X=CT*X1-ST*Y1
      Y= ST*X1+CT*Y1
      VCSTE1=X-XSAT
      VCSTE2=Y-YSAT
      VCSTE3=Z-ZSAT
      VCSES3=B31*VCSTE1+B32*VCSTE2+B33*VCSTE3
      ZNORM=SQRT(VCSTE1**2+VCSTE2**2+VCSTE3**2)
      X3=VCSES3/ZNORM
      UMV=ATAN2(X3,SQRT(RFACT-X3**2))-ROASIN
      XLIN=PICLIN-UMV/RADLIN
      PARLIN=IFIX(XLIN-1.0)/NUMSEN
      IF(I.EQ.2) GO TO 50
      SAMTIM=TIME2
      OLDLIN=XLIN
   50 CONTINUE
      SCNNUM=(IFIX(OLDLIN+XLIN)/2.0-1.0)/8.0
      SCNFRC=(SCNNUM-SCAN1)/(SCAN2-SCAN1)
      XLIN=OLDLIN+SCNFRC*(XLIN-OLDLIN)
      SAMTIM=TIME1+TMPSCL*(SCNNUM-SCAN1)
      CALL SATVEC(SAMTIM,XSAT,YSAT,ZSAT)
      COSA=X*XSAT+Y*YSAT+Z*ZSAT
      CTST=0.0001*R*XHT+RSQ
      IF(COSA.LT.CTST) NVXEAS=-1
      XSATS1=B11*XSAT+B12*YSAT+B13*ZSAT
      YSATS2=B21*XSAT+B22*YSAT+B23*ZSAT
      CT=COS(EMEGA*SAMTIM+XREF)
      ST=SIN(EMEGA*SAMTIM+XREF)
      X=CT*X1-ST*Y1
      Y= ST*X1+CT*Y1
      VCSTE1=X-XSAT
      VCSTE2=Y-YSAT
      VCSTE3=Z-ZSAT
      VCSES1=B11*VCSTE1+B12*VCSTE2+B13*VCSTE3
      VCSES2=B21*VCSTE1+B22*VCSTE2+B23*VCSTE3
      VCSES3=B31*VCSTE1+B32*VCSTE2+B33*VCSTE3
      XNORM=SQRT(ZNORM**2-VCSES3**2)
      YNORM=SQRT(XSATS1**2+YSATS2**2)
      ZNORM=SQRT(VCSTE1**2+VCSTE2**2+VCSTE3**2)
      X3=VCSES3/ZNORM
      UMV=ATAN2(X3,SQRT(RFACT-X3**2))-ROASIN
      SLIN=SIN(UMV)
      CLIN=COS(UMV)
      U=ROTM11*CLIN+ROTM13*SLIN
      V=ROTM21*CLIN+ROTM23*SLIN
      XELE=PICELE+ASIN((XSATS1*VCSES2-YSATS2*VCSES1)/(XNORM*YNORM))/RADE
     1LE
      XELE=XELE+ATAN2(V,U)/RADELE
      XELE=XELE-GAMMA-GAMDOT*SAMTIM
      RETURN
      END
C ********************************************************************
C
      FUNCTION NVXOPT(IFUNC,XIN,XOUT)
C IFUNC= 'SPOS'    SUBSATELLITE LAT/LON
C        XIN - NOT USED
C        XOUT - 1. SUB-SATELLITE LATITUDE
C             - 2. SUB-SATELLITE LONGITUDE
      INTEGER SPOS
      DATA SPOS/4HSPOS/
      REAL*4 XIN(*),XOUT(2)
      CHARACTER*4 CLIT,CFUNC
      CHARACTER*12 CFF
      NVXOPT=0


      IF(IFUNC.EQ.SPOS) THEN
         CALL SATPOS(0,ITIME(PICTIM),X,Y,Z)
         CALL NXYZLL(X,Y,Z,XOUT(1),XOUT(2))
      ELSE
         NVXOPT=1
      ENDIF
      RETURN
      END
C ********************************************************************
C
      SUBROUTINE SATPOS(INORB,NTIME,X,Y,Z)
C $ CALCULATE SATELLITE POSITION VECTOR FROM THE EARTH'S CENTER.
C $ INORB = (I) INPUT  INITIALIZATION FLAG - SHOULD = 0 ON FIRST CALL TO
C $   SATPOS, 1 ON ALL SUBSEQUENT CALLS.
C $ NTIME = (I) INPUT  TIME (HOURS, MINUTES, SECONDS) IN HHMMSS FORMAT
C $ X, Y, Z = (R) OUTPUT  COORDINATES OF POSITION VECTOR
C
      IMPLICIT REAL*8 (A-H,O-Z)
      REAL*4 DEGLIN,DEGELE,SPINRA,SEMIMA,OECCEN,ORBINC,PERHEL
      REAL*4 ASNODE,DECLIN,RASCEN,PICLIN,PRERAT,PREDIR,PITCH,YAW
      REAL*4 ROLL,SKEW
      REAL*4 X,Y,Z
      COMMON/NAVCOM/NAVDAY,LINTOT,DEGLIN,IELTOT,DEGELE,SPINRA,IETIMY,IET
     1IMH,SEMIMA,OECCEN,ORBINC,PERHEL,ASNODE,NOPCLN,DECLIN,RASCEN,PICLIN
     2,PRERAT,PREDIR,PITCH,YAW,ROLL,SKEW
C
      IF(INORB.NE.0)GO TO 1
      INORB=1
      PI=3.14159265D0
      RDPDG=PI/180.0
      RE=6378.388
      GRACON=.07436574D0
      SOLSID=1.00273791D0
      SHA=100.26467D0
      SHA=RDPDG*SHA
      IRAYD=74001
      IRAHMS=0
      O=RDPDG*ORBINC
      P=RDPDG*PERHEL
      A=RDPDG*ASNODE
      SO=SIN(O)
      CO=COS(O)
      SP=SIN(P)*SEMIMA
      CP=COS(P)*SEMIMA
      SA=SIN(A)
      CA=COS(A)
      PX=CP*CA-SP*SA*CO
      PY=CP*SA+SP*CA*CO
      PZ=SP*SO
      QX=-SP*CA-CP*SA*CO
      QY=-SP*SA+CP*CA*CO
      QZ=CP*SO
      SROME2=SQRT(1.0-OECCEN)*SQRT(1.0+OECCEN)
      XMMC=GRACON*RE*DSQRT(RE/SEMIMA)/SEMIMA
 1    DIFTIM=TIMDIF(IETIMY,IETIMH,NAVDAY,NTIME)
      XMANOM=XMMC*DIFTIM
      ECANM1=XMANOM
      EPSILN=1.0E-8
      DO 2 I=1,20
      ECANOM=XMANOM+OECCEN*DSIN(ECANM1)
      IF(DABS(ECANOM-ECANM1).LT.EPSILN)GO TO 3
 2    ECANM1=ECANOM
 3    XOMEGA=DCOS(ECANOM)-OECCEN
      YOMEGA=SROME2*DSIN(ECANOM)
      XS=XOMEGA*PX+YOMEGA*QX
      YS=XOMEGA*PY+YOMEGA*QY
      ZS=XOMEGA*PZ+YOMEGA*QZ
      DIFTIM=TIMDIF(IRAYD,IRAHMS,NAVDAY,NTIME)
      RA=DIFTIM*SOLSID*PI/720.0D0+SHA
      RAS=DMOD(RA,2.0*PI)
      CRA=COS(RAS)
      SRA=SIN(RAS)
      X=CRA*XS+SRA*YS
      Y=-SRA*XS+CRA*YS
      Z=ZS
      RETURN
      END
C ********************************************************************
C    From this point on, the routines included are the McIDAS
C subroutines called by the navigation routines to perform specific low
C level tasks.  The FORTRAN routines can be compiled along with the
C routines above.  The assembler routine must either be assembled or
C rewritten in FORTRAN.  The rewriting is not recommended.
C ********************************************************************
C
      FUNCTION ICON1(YYMMDD)
C     CONVERTS YYMMDD TO YYDDD
      IMPLICIT INTEGER(A-Z)
      DIMENSION NUM(12)
      DATA NUM/0,31,59,90,120,151,181,212,243,273,304,334/
C
      YEAR=MOD(YYMMDD/10000,100)
      MONTH=MOD(YYMMDD/100,100)
      DAY=MOD(YYMMDD,100)
      IF(MONTH.LT.0.OR.MONTH.GT.12)MONTH=1
      JULDAY=DAY+NUM(MONTH)
      IF(MOD(YEAR,4).EQ.0.AND.MONTH.GT.2) JULDAY=JULDAY+1
      ICON1=1000*YEAR+JULDAY
      RETURN
      END
C ********************************************************************
C
      SUBROUTINE EPOCH(IETIMY,IETIMH,SEMIMA,OECCEN,XMEANA)
C EPOCH  PHILLI 0173 NAV: FINDS TIME OF PERIGEE FROM KEPLERIAN EPOCH
      PARAMETER (PI=3.14159265)
      PARAMETER (RDPDG=PI/180.0)
      PARAMETER (RE=6378.388)
      PARAMETER (GRACON=0.07436574)
      LEAPYR(IY)=366-(MOD(IY,4)+3)/4
C
      XMMC=GRACON*SQRT(RE/SEMIMA)**3
      XMANOM=RDPDG*XMEANA
      TIME=(XMANOM-OECCEN*SIN(XMANOM))/(60.0*XMMC)
      TIME1=FLALO(IETIMH)
      TIME=TIME1-TIME
      IDAY=0
      IF(TIME.GT.48.0)GO TO 8
      IF(TIME.GT.24.0)GO TO 1
      IF(TIME.LT.-24.0)GO TO 2
      IF(TIME.LT.0.0)GO TO 3
      GO TO 4
 8    TIME=TIME-48.0
      IDAY=2
      GO TO 4
 1    TIME=TIME-24.0
      IDAY=1
      GO TO 4
 2    TIME=TIME+48.0
      IDAY=-2
      GO TO 4
 3    TIME=TIME+24.0
      IDAY=-1
 4    IETIMH=ITIME(TIME)
      IF(IDAY.EQ.0)RETURN
      JYEAR=MOD(IETIMY/1000,100)
      JDAY=MOD(IETIMY,1000)
      JDAY=JDAY+IDAY
      IF(JDAY.LT.1)GO TO 5
      JTOT=LEAPYR(JYEAR)
      IF(JDAY.GT.JTOT)GO TO 6
      GO TO 7
 5    JYEAR=JYEAR-1
      JDAY=LEAPYR(JYEAR)+JDAY
      GO TO 7
 6    JYEAR=JYEAR+1
      JDAY=JDAY-JTOT
 7    IETIMY=1000*JYEAR+JDAY
      RETURN
      END
C ********************************************************************
C
      FUNCTION RAERAC(IYRDY,IHMS,RAE)
C RAERAC PHILLI 0174 NAV: CONVERTS EARTH LON TO CELESTIAL LON
C INPUT PARAMETERS
C     IYRDY = YEAR AND JULIAN DAY (INTEGER YYDDD)
C     IHMS = TIME (INTEGER HHMMSS)
C     RAE = EARTH LONGITUDE (REAL*4 DEGREES)
C FUNCTION VALUE IS CELESTIAL LONGITUDE IN REAL*4 DEGREES
C REF: ESCOBAL, "METHODS OF ORBIT DETERMINATION." WILEY & SONS, 1965
C GREENWICH SIDEREAL TIME := ANGLE BETWEEN PRIME MERID. & 0 DEG R.A.
C JULIAN DATE := # DAYS ELAPSED SINCE 12 NOON ON JAN 1, 4713 B.C.
C APPROXIMATE FORMULA FOR GREENWICH SIDEREAL TIME AT 0Z:
C G.S.T (DEG) = S(0) = 99.6909833 + 36000.7689*C + 0.00038708*C*C ,WHERE
C    C = TIME IN CENTURIES = (J.D. - 2415020.0) / 36525
C FOR G.S.T. AT OTHER TIMES OF (SAME) DAY, USE:
C G.S.T AT TIME T = S(T) = S(0) + (T * DS/DT)
C DS/DT = 1 + (1 / 365.24219879) = 1.00273790927 REVOLUTIONS/DAY
C       = 0.250684477 DEGREES/MINUTE
C FROM TABLE, J.D. AT 0Z ON JAN 1,1974 = 2442048.5
C THEN S(0) AT 0Z ON JAN 1,1974 = 100.2601800
C
      DOUBLE PRECISION TIMDIF,RAHA,SOLSID,SHA
      SHA=100.26467D0
      IRAYD=74001
      IRAHMS=0
      SOLSID=1.00273791D0
      RAHA=RAE+TIMDIF(IRAYD,IRAHMS,IYRDY,IHMS)*SOLSID/4.0D0+SHA
      RAC=DMOD(RAHA,360.0D0)
      IF(RAC.LT.0.0)RAC=RAC+360.0
      RAERAC=RAC
      RETURN
      END
C ********************************************************************
C
      FUNCTION RACRAE(IYRDY,IHMS,RAC)
C RACRAE PHILLI 0174 NAV: CONVERTS CELESTIAL ONG TO EARTH LON
C INPUT PARAMETERS--
C     IYRDY = YEAR AND JULIAN DAY (INTEGER YYDDD)
C     IHMS = TIME (INTEGER HHMMSS)
C     RAC = CELESTIAL LONGITUDE (REAL*4 DEGREES)
C FUNCTION VALUE IS EARTH LONGITUDE IN REAL*4 DEGREES
C
      DOUBLE PRECISION TIMDIF,RAHA,SOLSID,SHA
      SHA=100.26467D0
      IRAYD=74001
      IRAHMS=0
      SOLSID=1.00273791D0
      RAHA=RAC-SHA+TIMDIF(IYRDY,IHMS,IRAYD,IRAHMS)*SOLSID/4.0D0
      RAE=DMOD(RAHA,360.0D0)
      IF(RAE.LT.0.0)RAE=RAE+360.0
      RACRAE=RAE
      RETURN
      END
C ********************************************************************
C
      FUNCTION TIMDIF(IYRDA1,IHMS1,IYRDA2,IHMS2)
C TIME DIFFERENCE IN MINUTES
C INPUTS (ALL INTEGER)
C IYRDA1 FIRST YEAR AND JULIAN DAY (YYDDD)
C IHMS1 FIRST TIME (HHMMSS)
C IYRDA2 SECOND YEAR AND DAY (YYDDD)
C IHMS2 SECOND TIME (HHMMSS)
C FUNC VAL (REAL*8) IS TIME DIFFERENCE IN MINUTES (POSITIVE IF
C FIRST DAY/TIME IS THE EARLIER OF THE TWO).
C
      DOUBLE PRECISION TIMDIF,D1,D2,T1,T2
      IY1=MOD(IYRDA1/1000,100)
      ID1=MOD(IYRDA1,1000)
      IFAC1=(IY1-1)/4+1
      D1=365*(IY1-1)+IFAC1+ID1-1
      IY2=MOD(IYRDA2/1000,100)
      ID2=MOD(IYRDA2,1000)
      IFAC2=(IY2-1)/4+1
      D2=365*(IY2-1)+IFAC2+ID2-1
      T1=1440.D0*D1+60.D0*FLALO(IHMS1)
      T2=1440.D0*D2+60.D0*FLALO(IHMS2)
      TIMDIF=T2-T1
      RETURN
      END
C ********************************************************************
C
C VECTOR EARTH-CENTER-TO-SAT (FUNC OF TIME)
      SUBROUTINE SATVEC(SAMTIM,X,Y,Z)
      DOUBLE PRECISION TWOPI,PI720,DE,TE,DRA,TRA,DNAV,TDIFRA,TDIFE
      DOUBLE PRECISION PI,RDPDG,RE,GRACON,SOLSID,SHA
      DOUBLE PRECISION DIFTIM,ECANM1,ECANOM,RA,XMANOM,TIMDIF
      DOUBLE PRECISION DABS,DMOD,DSQRT,DSIN,DCOS,DATAN2
      COMMON/NAVCOM/NAVDAY,LINTOT,DEGLIN,IELTOT,DEGELE,SPINRA,IETIMY,IET
     1IMH,SEMIMA,OECCEN,ORBINC,PERHEL,ASNODE,NOPCLN,DECLIN,RASCEN,PICLIN
     2,PRERAT,PREDIR,PITCH,YAW,ROLL,SKEW
      DATA NAVSAV/0/
      IF(NAVDAY.EQ.NAVSAV) GO TO 10
      NAVSAV=NAVDAY
      PI=3.14159265D0
      TWOPI=2.0*PI
      PI720=PI/720.
      RDPDG=PI/180.0
      RE=6378.388
      GRACON=.07436574D0
      SOLSID=1.00273791D0
      SHA=100.26467D0
      SHA=RDPDG*SHA
      IRAYD=74001
      IRAHMS=0
      O=RDPDG*ORBINC
      P=RDPDG*PERHEL
      A=RDPDG*ASNODE
      SO=SIN(O)
      CO=COS(O)
      SP=SIN(P)*SEMIMA
      CP=COS(P)*SEMIMA
      SA=SIN(A)
      CA=COS(A)
      PX=CP*CA-SP*SA*CO
      PY=CP*SA+SP*CA*CO
      PZ=SP*SO
      QX=-SP*CA-CP*SA*CO
      QY=-SP*SA+CP*CA*CO
      QZ=CP*SO
      SROME2=SQRT(1.0-OECCEN)*SQRT(1.0+OECCEN)
      XMMC=GRACON*RE*DSQRT(RE/SEMIMA)/SEMIMA
      IEY=MOD(IETIMY/1000,100)
      IED=MOD(IETIMY,1000)
      IEFAC=(IEY-1)/4+1
      DE=365*(IEY-1)+IEFAC+IED-1
      TE=1440.0*DE+60.0*FLALO(IETIMH)
      IRAY=IRAYD/1000
      IRAD=MOD(IRAYD,1000)
      IRAFAC=(IRAY-1)/4+1
      DRA=365*(IRAY-1)+IRAFAC+IRAD-1
      TRA=1440.0*DRA+60.0*FLALO(IRAHMS)
      INAVY=MOD(NAVDAY/1000,100)
      INAVD=MOD(NAVDAY,1000)
      INFAC=(INAVY-1)/4+1
      DNAV=365*(INAVY-1)+INFAC+INAVD-1
      TDIFE=DNAV*1440.-TE
      TDIFRA=DNAV*1440.-TRA
      EPSILN=1.0E-8
   10 TIMSAM=SAMTIM*60.0
      DIFTIM=TDIFE+TIMSAM
      XMANOM=XMMC*DIFTIM
      ECANM1=XMANOM
      DO 2 I=1,20
      ECANOM=XMANOM+OECCEN*DSIN(ECANM1)
      IF(DABS(ECANOM-ECANM1).LT.EPSILN)GO TO 3
 2    ECANM1=ECANOM
 3    XOMEGA=DCOS(ECANOM)-OECCEN
      YOMEGA=SROME2*DSIN(ECANOM)
      Z =XOMEGA*PZ+YOMEGA*QZ
      Y =XOMEGA*PY+YOMEGA*QY
      X =XOMEGA*PX+YOMEGA*QX
      RETURN
      END
C ********************************************************************
C
      SUBROUTINE NLLXYZ(XLAT,XLON,X,Y,Z)
C-----CONVERT LAT,LON TO EARTH CENTERED X,Y,Z
C     (DALY, 1978)
C-----XLAT,XLON ARE IN DEGREES, WITH NORTH AND WEST POSITIVE
C-----X,Y,Z ARE GIVEN IN KM. THEY ARE THE COORDINATES IN A RECTANGULAR
C        FRAME WITH ORIGIN AT THE EARTH CENTER, WHOSE POSITIVE
C        X-AXIS PIERCES THE EQUATOR AT LON 0 DEG, WHOSE POSITIVE Y-AXIS
C        PIERCES THE EQUATOR AT LON 90 DEG, AND WHOSE POSITIVE Z-AXIS
C        INTERSECTS THE NORTH POLE.
      DATA ASQ/40683833.48/,BSQ/40410330.18/,AB/40546851.22/
      DATA RDPDG/1.7453292E-02/,PI/3.1415926/
      YLAT=RDPDG*XLAT
C-----CONVERT TO GEOCENTRIC (SPHERICAL) LATITUDE
      YLAT=ATAN2(BSQ*SIN(YLAT),ASQ*COS(YLAT))
      YLON=-RDPDG*XLON
      SNLT=SIN(YLAT)
      CSLT=COS(YLAT)
      CSLN=COS(YLON)
      SNLN=SIN(YLON)
      TNLT=(SNLT/CSLT)**2
      R=AB*SQRT((1.0+TNLT)/(BSQ+ASQ*TNLT))
      X=R*CSLT*CSLN
      Y=R*CSLT*SNLN
      Z=R*SNLT
      RETURN
      END
C ********************************************************************
C
      SUBROUTINE NXYZLL(X,Y,Z,XLAT,XLON)
C-----CONVERT EARTH-CENTERED X,Y,Z TO LAT & LON
C-----X,Y,Z ARE GIVEN IN KM. THEY ARE THE COORDINATES IN A RECTANGULAR
C        COORDINATE SYSTEM WITH ORIGIN AT THE EARTH CENTER, WHOSE POS.
C        X-AXIS PIERCES THE EQUATOR AT LON 0 DEG, WHOSE POSITIVE Y-AXIS
C        PIERCES THE EQUATOR AT LON 90 DEG, AND WHOSE POSITIVE Z-AXIS
C        INTERSECTS THE NORTH POLE.
C-----XLAT,XLON ARE IN DEGREES, WITH NORTH AND WEST POSITIVE
      DATA RDPDG/1.745329252E-2/
      DATA ASQ/40683833.48/,BSQ/40410330.18/,AB/40546851.22/
C
      XLAT=100.0
      XLON=200.0
      IF(X.EQ.0..AND.Y.EQ.0..AND.Z.EQ.0.) GO TO 90
      A=ATAN(Z/SQRT(X*X+Y*Y))
C-----CONVERT TO GEODETIC LATITUDE
      XLAT=ATAN2(ASQ*SIN(A),BSQ*COS(A))/RDPDG
      XLON=-ATAN2(Y,X)/RDPDG
   90 RETURN
      END
C ********************************************************************
C
      FUNCTION FLALO(M)
C INTEGER ANGLE (FORMAT DDDMMSS) OR TIME (FORMAT HHMMSS) TO REAL*4
C M=ANGLE OR TIME
C
      N=IABS(M)
 2    FLALO=FLOAT(N/10000)+FLOAT(MOD(N/100,100))/60.0+FLOAT(MOD(N,100))/
     *3600.0
      IF (M.LT.0) FLALO=-FLALO
      RETURN
      END
C ********************************************************************
C
      FUNCTION ITIME(X)
C $ FLOATING POINT TIME TO PACKED INTEGER ( SIGN HH MM SS )
C $ X = (R) INPUT  FLOATING POINT TIME
C
      IF(X.LT.0.0)GO TO 1
      Y=X
      I=1
      GO TO 2
 1    Y=-X
      I=-1
 2    J=3600.0*Y+0.5
      ITIME=10000*(J/3600)+100*MOD(J/60,60)+MOD(J,60)
      ITIME=I*ITIME
      RETURN
      END
C ********************************************************************
C
