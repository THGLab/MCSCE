SUBROUTINE ATAREA(NT,XYZ,NI2,NI3,RADN, RADP,TYPE,AREAS,TAREA, TIME,NOHY) 
    C This subroutine computes the approximate area of each 
    C atom in the molecule(s) described in the argument list 
    C Input: NT - 1"4, total number of atoms 
    C XYZ(3,1000) - R*4, coordinates of each atom in Angstroms. 
    C N12(6,1000) - 1"2, atom numbers of all atoms which are 1,2 
    C (directly bound) to each atom. Dimensioning 
    C assumes up to 6 attachments for each atom. 
    C N13(30,1000) - 1"2, atom numbers of all atoms which are 1,3 to each 
    C atom. Dimensioned for up to 6 attachments per atom. 
    C RADN(1000) - R*4, van der Waals radius of each atom in Angstroms. 
    C RADP - R*4, radius of solvent probe in Angstroms. 
    C TYPE(1000) - 1"2, atom types of all atoms 
    C Output: AREAS(1000) - R*4, area of each atom in square Angstroms 
    C TAREA - R*4, total Area of the molecule atom. 
    REAL*4 XYZ(3,1000),RADN(1000),AREAS(1000),TYPEPAR(25), 
    1 BMLT,PARAM(3),RADP, 
    2 PROBD, PI,RADNP, S,DELX,DELY, DELZ, RNI, 
    3 DNISQ,DENOM, DIST, BEE,PROBABILITY,ASURF,ASURFT, 
    4 TAREA, START 
    INTEGER*2 NI2(6,1000),NI3(30,1000),TYPE(1000), 
    1 MAXNI2(1000),MAXNI3(1000) 
    INTEGER*4 NT,N,I 
    LOGICAL NOHY 
    C *** Load parameters depending on connectivity 
    C *** 1,2 1,3 1,4 
    DATA PARAM/.8875,0.3516,0.3156/ 
    C *** Load atom type parameters 
    C *** C(sp3) 
    DATA TYPEPAR/ 2.149 
    C *** C(sp2) 
    1 1.554, 
    CH(sp3) CH2(SP3) CH3(sp3) 
    1.276, 1.045, .88, 
    CH(sp2) CH2(sp2) C(sp) 
    1.073, .961, .737, 
    H(NH) H(NH+) H(SH) 
    1.128, 1.049, .928, 
    O(sp2) 0 - N(sp3) 
    .926, .922, 1.215, 
    NH, =N-(sp2) S F 
    1.028, 1.121, .958, 
    Br I~ P 
    .898, .876, 1.5887, 1.0/ 
    C *** H (OH) 
    2 .944, 
    C *** 0 (sp3) 
    3 1.08 
    C *** N (sp2) 
    4 1.413 
    C *** C1 
    5 .958, 
    I0 
    15 
    20 
    25 
    C 
    & 
    C 
    *** Save maximum extent of NI2 and NI3 arrays *** 
    DO 25 I = I,NT 
    DO i0 J = I, 6 
    IF (NI2(J,I).EQ.0) GO TO 15 
    MAXNI2 (I) = J 
    DO 20 J = 1,30 
    IF (NI3(J,I).EQ.0) GO TO 25 
    MAXNI3(I) = J 
    CONTINUE 
    *** Begin area calculation *** 
    START = CPU TIME(0.0) 
    PROBD = 2.0~RADP 
    PI = ACOS(-I.) 
    ASURFT = 0.0 
    DO 200 N = I,NT 
    AREAS(N) = 0.0 
    IF (TYPE(N) .EQ.25) WRITE (6,*) 
    '*** No parameter p(i) for atom # ',N 
    *** Skip hydrogens if NOHY=.TRUE. *** 
    IF (NOHY.AND.TYPE(N) .GE.9.AND.TYPE(N).LE.12) GO TO 200 
    RADNP = RADN (N) +RADP 
    C 
    C 
    C 
    C 
    70 
    C 
    75 
    80 
    C 
    i00 
    C 
    C 
    200 
    C 
    C 
    *** Calculate surface area with no overlap *** 
    S = 4. *P I*RADNP*RADNP 
    *** Initialize probability *** 
    PROBABILITY = i. 0 
    *** Loop over all atoms to find intersections and areas *** 
    DO i00 I=I,NT 
    *** Skip hydrogens if NOHY=.TRUE. *** 
    IF (NOHY.AND.TYPE(I).GE.9.AND.TYPE(I).LE.12) GO TO i00 
    IF (I.EQ.N) GO TO 100 
    *** Find distance between atoms *** 
    DELX = XYZ(I,N)-XYZ(I,I) 
    DELY = XYZ(2,N)-XYZ(2,I) 
    DELZ = XYZ(3,N)-XYZ(3, I) 
    DNISQ = DELX*DELX + DELY*DELY + DELZ*DELZ 
    RNI = RADN (I) +RADN (N) +PROBD 
    IF(DNISQ.GE.RNI*RNI) GO TO 100 !no overlap of N and I 
    BMLT = PARAM (3) 
    *** If distance too long for 1,2- or 1,3-interactions skip checking *** 
    IF(DNISQ.GT.12.25) GO TO 80 
    *** Check for 1,2-interaction of atoms N and I *** 
    DO 70 J = I,MAXNI2(I) 
    IF (NI2(J,I) .EQ.N) THEN 
    BMLT = PARAM(1) 
    GO TO 80 
    ENDIF 
    CONTINUE 
    *** Check for 1,3-interactions *** 
    DO 75 J=I,MAXNI3(I) 
    IF (NI3(J,I) .EQ.N) THEN 
    BMLT = PARAM(2) 
    GO TO 80 
    ENDIF 
    CONTINUE 
    CONTINUE 
    BMLT = BMLT * TYPEPAR(TYPE (N)) 
    *** Calculate excluded surface area *** 
    DIST = SQRT(DNISQ) 
    BEE = BMLT*PI*RADNP* (RNI-DIST) * (I.+ (RADN (I) -RADN (N)) /DIST) 
    *** Calculate overlap probabilty product *** 
    PROBABILITY = PROBABILITY* (I. -BEE/S) 
    CONTINUE 
    *** All done, so calculate accessible area *** 
    ASURF = S'PROBABILITY 
    IF (ASURF.LT.0.) ASURF=0. 
    ASURFT=ASURFT + ASURF 
    *** Areas(N) is approximate area of atom N *** 
    AREAS (N) = ASURF 
    CONTINUE 
    *** TAREA, total area, is sum of AREAS() *** 
    TAREA = ASURFT 
    *** Save total cpu time *** 
    TIME = CPU TIME(START) 
    RETURN 
    END 