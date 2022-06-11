import math
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

def ElementsKeplerians(r,v):
    '''
    Parameters
    ----------
    r : Array de les coordenades cartesianes posició (x,y,z) en metres
    v : Array de les coordenades cartesianes velocitat en metres per segon

    Returns
    -------
    a : Semieix major en metres
    ecc : Eccentricitat
    i : Inclinacií en graus
    O : Longitud del node ascendent en graus
    w : Longitud del periàpside en graus
    nu : Anomalia verdadera en graus
        
    !!! COMPTE !!!
    No estan protegides condicions com i=0 on certes magnituds no estan
    ben definides i el programa pot fallar.
    '''
    mu = 3.986004418E14
    h = np.cross(r,v)
    h_norm = math.sqrt(h[0]**2 + h[1]**2 + h[2]**2)
    i = math.acos(h[2]/h_norm)
    r_norm = math.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
    v_norm = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    e = (np.cross(v,h)/mu) - r/r_norm
    ecc = math.sqrt(e[0]**2 + e[1]**2 + e[2]**2)
    a = (r_norm*mu)/(2*mu - r_norm*v_norm**2)
    n = math.sqrt(h[0]**2 + h[1]**2)
    if np.dot(r,v)<0:
        nu = 2*math.pi - math.acos(np.dot(e,r)/(ecc*r_norm))
    else:
        nu = math.acos(np.dot(e,r)/(ecc*r_norm))
    if h[0]>=0:
        O = math.acos(-h[1]/n)
    else:
        O = 2*math.pi - math.acos(-h[1]/n)
    if e[2]>=0:
        w = math.acos((-h[1]*e[0] + h[0]*e[1])/(n*ecc))
    else:
        w = 2*math.pi - math.acos((-h[1]*e[0] + h[0]*e[1])/(n*ecc))
    i = i*180/math.pi
    O = O*180/math.pi
    w = w*180/math.pi
    nu = nu*180/math.pi
    return a,ecc,i,O,w,nu

def ElementsCartesians(a,e,i,O,w,nu):
    '''
    Parameters
    ----------
    a : Semieix major en metres
    ecc : Eccentricitat
    i : Inclinacií en graus
    O : Longitud del node ascendent en graus
    w : Longitud del periàpside en graus
    nu : Anomalia verdadera en graus

    Returns
    -------
    pos : Array de les coordenades cartesianes posició (x,y,z) en metres
    vel : Array de les coordenades cartesianes velocitat en metres per segon
        
    !!! COMPTE !!!
    No estan protegides condicions com i=0 on certes magnituds no estan
    ben definides i el programa pot fallar.
    '''
    i = i*math.pi/180
    # No sé el perquè dels 180 de dif. però és així
    O = 180 - O
    w = 180 - w
    O = O*math.pi/180
    w = w*math.pi/180
    nu = nu*math.pi/180
    mu = 3.986004418E14
    r = (a*(1 - e**2))/(1 + e*math.cos(nu))
    E = math.acos((1 - r/a)/e)
    #posicio i velocitat en el pla orbital
    #z perpendicular a l'eix orbital
    #x apuntant al periàpside
    ox = r*math.cos(nu)
    oy = r*math.sin(nu)
    oz = 0
    ct = math.sqrt(mu*a)/r
    opx = -ct*math.sin(E)
    opy = ct*math.sqrt(1 - e**2)*math.cos(E)
    opz = 0
    o = np.array([ox,oy,oz])
    op = np.array([opx,opy,opz])
    R_z = np.array([[math.cos(-O),-math.sin(-O),0],
                    [math.sin(-O),math.cos(-O),0],
                    [0,0,1]])
    R_x = np.array([[1,0,0],
                    [0,math.cos(-i),-math.sin(-i)],
                    [0,math.sin(-i),math.cos(-i)]])
    R_z2 = np.array([[math.cos(-w),-math.sin(-w),0],
                    [math.sin(-w),math.cos(-w),0],
                    [0,0,1]])
    rot1 = np.matmul(R_z,R_x)
    rot2 = np.matmul(rot1,R_z2)
    pos = np.matmul(rot2,o)
    vel = np.matmul(rot2,op)
    return pos,vel

def observador(phi,theta,H):
    '''
    Parameters
    ----------
    phi : Latitud geodètica en graus de l'observador
    theta : Temps sideral de l'observador (longitud amb la rotació pertinent)
            també en graus
    H : Altitud respecte el nivell del mar de l'observador en metres

    Returns
    -------
    x,y,z : Coordenades cartesianes respecte el centre de la Terra (0,0,0)
            de l'observador en metres
    '''
    phi = phi*math.pi/180
    theta = theta*math.pi/180
    #Aplatament, considero terra asfèrica
    f = 1/298.3
    e = math.sqrt(2*f - f**2)
    ae = 6378136.6
    D = math.sqrt(1 - (e*math.sin(phi))**2)
    G1 = (ae/D) + H
    G2 = ((ae*((1 - f)**2))/D) + H
    xc = G1*math.cos(phi)
    zc = G2*math.sin(phi)
    x = xc*math.cos(theta)
    y = xc*math.sin(theta)
    z = zc
    return x,y,z

def JD(data):
    '''
    Parameters
    ----------
    data : Data d'observació en el calendari Gregorià i hora terrestre
           format : datetime(any,mes,dia,hora,minut,second,microsegons)

    Returns
    -------
    dies : Data Juliana
    '''
    # ANYS
    anys = data.year - 1 + 4713
    traspas = 0
    for i in range(-4713,data.year):
        if i<1582:
            if i%4==0:
                traspas += 1
        if i>=1582:
            if i%4==0:
                traspas += 1
                if i%100==0:
                    traspas -= 1
                    if i%400==0:
                        traspas += 1
    dies = 365*anys + traspas
    #Canvi al calendari Gregorià 4/10/1582 -> 15/10/1582
    if data.year>1582:
        dies -= 10
    if data.year==1582 and data.month>10:
        dies -= 10
    if data.year==1582 and data.month==10 and data.day>4:
        dies -= 10
    
    #DIES
    dies += data.day - 1
    
    #HORA EXACTA
    dies += ((data.hour-12)/24 + (data.minute)/1440
            + data.second/86400 + data.microsecond/86400000000)
    
    #MESOS
    mes = data.month
    af = 0
    if data.month>2:
        if data.year<1582:
            if data.year%4==0:
                af +=1
        if data.year>=1582:
            if data.year%4==0:
                af +=1
                if data.year%100==0:
                    af -=1
                    if data.year%400==0:
                        af +=1
    if mes==2 or mes==6 or mes==7:
        af = 1
    elif mes==3:
        af = -1
    elif mes==8:
        af = 2
    elif mes==9 or mes==10:
        af = 3
    elif mes==11 or mes==12:
        af = 4
    dies += (mes-1)*30 + af
    
    return dies

def sideraltime(dataJ,long,t):
    '''
    Parameters
    ----------
    data : Data Juliana
    long : Longitud est des del meridià de Greenwich de l'observador
    t : Moment posterior a la data Juliana sobre el qual es vol saber el temps 
        sideral en segons

    Returns
    -------
    theta : Temps sideral en graus
    '''
    T_u = (dataJ - 2415020)/36525
    theta_g0 = 99.6909833 + 36000.7689*T_u + 0.00038708*T_u**2
    if theta_g0>(360):
        voltes = theta_g0/(360)
        theta_g0 = theta_g0 - int(voltes)*360
    if theta_g0<0:
        voltes = theta_g0/(360)
        theta_g0 = theta_g0 + int(voltes)*360
    
    theta = theta_g0 + (t-dataJ)*0.004178074 + long
    
    if theta>(360):
        voltes = theta/(360)
        theta = theta - int(voltes)*360
    if theta<0:
        voltes = theta/(360)
        theta = theta + int(voltes)*360
    return theta

def RA_dec(x,y,z):
    '''
    Parameters
    ----------
    x,y,z : Coordenades cartesianes geocèntriques deL satèl·lit en metres
            considerant el centre de la Terra com el punt (0,0,0)

    Returns
    -------
    RA : Ascensió recta en hores (si x=y=0 retorna un missatge d'error i pren RA=0)
    dec : Declinació en graus (si x=y=z=0 retorna un missatge d'error i pren dec=0)
    '''
    # DECLINACIÓ
    if (x**2 + y**2)==0:
        if z>0:
            dec = 90
        elif z<0:
            dec = -90
        elif x==0 and y==0 and z==0:
            print("Singularitat de coordenades. Què fas aquí?")
            dec = 0
    else:
        dec = (math.atan(z/(math.sqrt(x**2 + y**2))))*180/math.pi
    # ASCENSIÓ RECTA
    if x==0 and y==0:
        print('Ascensió recta indeterminada.')
        RA = 0 
    elif x==0 and y>0:
        RA = 90
    elif x==0 and y<0:
        RA = -90
    elif y==0 and x>0:
        RA=0
    elif y==0 and x<0:
        RA = 180
    elif x>0 and y>0:
        RA = math.atan(y/x)*180/math.pi  
    elif x<0 and y>0:
        RA = math.atan(-x/y)*180/math.pi + 90
    elif x<0 and y<0:
        RA = math.atan(y/x)*180/math.pi + 180
    elif x>0 and y<0:
        RA = math.atan(x/-y)*180/math.pi + 270
    RA = RA*24/360
    return RA,dec

def A_h(x,y,z,xo,yo,zo):
    '''
    Aquesta funció retorna l'azimut i elevació d'un objecte respecte l'observador.

    Parameters
    ----------
    x,y,z : Coordenades de l'objecte en metres
    xo,yo,zo : Coordenades de l'observador en metres

    Returns
    -------
    A : Azimut en graus
    h : Elevació en graus respecte l'horitzó de l'observador

    '''
    f = 1/298.3
    e = math.sqrt(2*f - f**2)
    ae = 6378136.6
    PNz = ae*math.sqrt(1-e**2)
    PN = np.array([-xo,-yo,PNz-zo])
    vx = x - xo
    vy = y - yo
    vz = z - zo
    sat_obs = np.array([vx,vy,vz])
    theta,phi = RA_dec(xo,yo,zo)
    theta = theta*math.pi/12
    phi = (90 - phi)*math.pi/180
    R_z = np.array([[math.cos(-theta),-math.sin(-theta),0],
                    [math.sin(-theta),math.cos(-theta),0],
                    [0,0,1]])
    R_y = np.array([[math.cos(-phi),0,math.sin(-phi)],
                    [0,1,0],
                    [-math.sin(-phi),0,math.cos(-phi)]])
    Rot = np.matmul(R_y,R_z)
    sor = np.matmul(Rot,sat_obs)
    RA,h = RA_dec(sor[0],sor[1],sor[2])
    Nord = np.matmul(Rot,PN)
    #Vectors al pla de l'observador:
    obj = np.array([sor[0],sor[1]])
    obj_norm = math.sqrt(sor[0]**2 + sor[1]**2)
    Ref = np.array([Nord[0],Nord[1]])
    Ref_norm = np.array([Nord[0]**1 + Nord[1]**2])
    A = math.acos((np.dot(obj,Ref))/(obj_norm*Ref_norm))
    A = A*180/math.pi
    if A<0:
        A = -A + 180
    return A,h

def EulerSimple(r,temps):
    '''
    Aquest script soluciona numèricament mitjançant el mètode d'Euler millorat
    les equacions del moviment per un sol cos.

    Parameters
    ----------
    r : array amb les tres coordenades de posició i velocitat en SI
        (x,y,z,vx,vy,vz)
    temps : Quants segons es vol simular el moviment de l'objecte
            !!! Cal tenir en compte que el nombre de passos en l'integració és
            sempre el mateix, 100000, i per tant, amb temps molt grans, la
            solució pot ser incorrecta.

    Returns
    -------
    x : Array amb totes les posicions x successives de l'objecte (100001) en m.
    y : Array amb totes les posicions y successives de l'objecte (100001) en m.
    z : Array amb totes les posicions z successives de l'objecte (100001) en m.
    t : Array amb tots els instants de temps al quals correponen les posicions
        de les altres arrays (100001) en segons
    impacte : Boolean
                Retorna True si l'òrbita ha impactat amb la superfície de la 
                Terra (aquí considerada esfèrica i de radi equatorial).
                Retorna False si l'òrbita no ha impactat amb la Terra

    '''
    x0 = r[0]
    y0 = r[1]
    z0 = r[2]
    vx0 = r[3]
    vy0 = r[4]
    vz0 = r[5]
    
    mu = 3.9860044188e14
    radi = 6378136.6
    
    passos = 100000
    deltat = temps/passos
    
    impacte = False

    x = ([x0])
    y = ([y0])
    z = ([z0])
    temps = 0
    t = ([0])

    for i in range(0,passos):
        if i==0:
            x1 = x0 + deltat*vx0
            y1 = y0 + deltat*vy0
            z1 = z0 + deltat*vz0
            vx1 = vx0 - deltat*mu*x0/(math.sqrt((x0**2+y0**2+z0**2))**3)
            vy1 = vy0 - deltat*mu*y0/(math.sqrt((x0**2+y0**2+z0**2))**3)
            vz1 = vz0 - deltat*mu*z0/(math.sqrt((x0**2+y0**2+z0**2))**3)
            temps = deltat
        
        x2 = x0 + 2*deltat*vx1
        y2 = y0 + 2*deltat*vy1
        z2 = z0 + 2*deltat*vz1
        vx2 = vx0 - 2*deltat*mu*x1/(math.sqrt((x1**2+y1**2+z1**2))**3)
        vy2 = vy0 - 2*deltat*mu*y1/(math.sqrt((x1**2+y1**2+z1**2))**3)
        vz2 = vz0 - 2*deltat*mu*z1/(math.sqrt((x1**2+y1**2+z1**2))**3)
        temps += deltat
        t = np.append(t,temps)
        
        x = np.append(x,x2)
        y = np.append(y,y2)
        z = np.append(z,z2)
        
        x0 = x1
        y0 = y1
        z0 = z1
        vx0 = vx1
        vy0 = vy1
        vz0 = vz1
        x1 = x2
        y1 = y2
        z1 = z2
        vx1 = vx2
        vy1 = vy2
        vz1 = vz2
        
        if (math.sqrt(x[-1]**2+y[-1]**2+z[-1]**2))<radi:
            impacte = True
            return x,y,z,t,impacte
            break
    return x,y,z,t,impacte

def f_series(r,rp,t):
    #Variables reduïdes
    mu = 1
    u = mu/(r**3)
    '''
    p = rp/r
    q = (rp**2 - u*r)/(r**2)
    funcio = (1 - 0.5*u*t**2 + 0.5*u*p*t**3 + ((3*u*q - 15*u*p**2 + 
            u**2)*t**4)/24 + ((7*u*p**3 - 3*u*p*q - p*u**2)*t**5)/8 + 
            ((630*u*p**2*q - 24*u**2*q - u**3 - 45*u*q**2 - 945*u*p**4 + 
            210*u**2*p**2)*t**6)/720 + ((882*u**2*p*q - 3150*u**2*p**3 - 
            9450*u*p**3*q + 1575*u*p*q**2 + 63*u**3*p + 
            10395*u*p**5)*t**7)/5040 + ((1107*u**2*q**2 - 24570*u**2*p**2*q - 
            2205*u**3*p**2 + 51975*u**2*p**4 - 42525*u*p**2*q**2 + 
            155925*u*p**4*q + 1575*u*q**3 + 117*u**3*q - 135135*u*p**6 + 
            u**4)*t**8)/40320)
    '''
    #El desenvolupament del mètode de Gauss es fa amb el primer ordre
    funcio = 1 - 0.5*u*t**2
    return funcio

def g_series(r,rp,t):
    #Variables reduïdes
    mu = 1
    u = mu/(r**3)
    '''
    p = rp/r
    q = (rp**2 - u*r)/(r**2)
    funcio = (t - (u*t**3)/6 + (u*p*t**4)/4 + ((9*u*q - 45*u*p**2 + 
            u**2)*t**5)/120 + ((210*u*p**3 - 90*u*p*q - 15*u**2*p)*t**6)/360 + 
            ((3150*u*p**2*q - 54*u**2*q - 225*u*q**2 - 4725*u*p**4 + 
            630*u**2*p**2 - u**3)*t**7)/5040 + ((3024*u**2*p*q - 
            12600*u**2*p**3 - 56700*u*p**3*q + 9450*u*p*q**2 + 62370*u*p**5 + 
            126*u**3*p)*t**8)/40320)
    '''
    #El desenvolupament del mètode de Gauss es fa amb el primer ordre
    funcio = t - (u*t**3)/6
    return funcio
 
def Gauss(t,alpha,delta,phi,lambda_E,H):
    '''
    Parameters
    ----------
    t : Tres instants d'observació en segons
    alpha : Tres ascensions rectes topocèntriques d'observació en hores
    delta : Tres declinacions topocèntriques d'observació en graus
    phi : Tres latituds geodètiques en graus dels tres observadors
    lambda_E : Tres longituds est en graus dels tres observadors
    H : Tres altituds dels observadors en metres

    Returns
    -------
    r_2 : Posició en metres del segon instant d'observació
    rpunt_2 : Velocitat en metres per segon del segon instant d'observació
    '''
    t0 = t[1]
    alpha = alpha*math.pi/12
    delta = delta*math.pi/180
    lambda_E = lambda_E*math.pi/180
    phi = phi*math.pi/180
    #Constants en unitats reduïdes (masses i radis equatorials terrestres)
    mu = 1
    k = 0.07436574/60

    #CÀLCUL
    #Variables temporals modificades
    tau_1 = k*(t[0]-t0)
    tau_3 = k*(t[2]-t0)
    tau_13 = tau_3 - tau_1
    A_1 = tau_3/tau_13
    B_1 = (tau_13**2-tau_3**2)*A_1/6
    A_3 = -tau_1/tau_13
    B_3 = (tau_13**2-tau_1**2)*A_3/6
    
    #Vectors normalitzats de coordenades topocèntriques de l'objecte
    L = np.empty([3,3])
    #Vectors de posició de l'observador
    R = np.empty([3,3])

    theta = np.empty([3])

    for i in range(0,3):
        L[0,i]=math.cos(delta[i])*math.cos(alpha[i])
        L[1,i]=math.cos(delta[i])*math.sin(alpha[i])
        L[2,i]=math.sin(delta[i])
        theta[i] = sideraltime(t0,lambda_E[i],t[i])
        R[0,i],R[1,i],R[2,i]=observador(phi[i],theta[i],H[i])
        R[0,i],R[1,i],R[2,i]=-R[0,i]/6378136.6,-R[1,i]/6378136.6,-R[2,i]/6378136.6
    
    #Inversa de la matriu L
    D = np.linalg.det(L)
    a_11 = (L[1,1]*L[2,2]-L[1,2]*L[2,1])/D
    a_12 = -(L[0,1]*L[2,2]-L[0,2]*L[2,1])/D
    a_13 = (L[0,1]*L[1,2]-L[0,2]*L[1,1])/D
    a_21 = -(L[1,0]*L[2,2]-L[1,2]*L[2,0])/D
    a_22 = (L[0,0]*L[2,2]-L[0,2]*L[2,0])/D
    a_23 = -(L[0,0]*L[1,2]-L[0,2]*L[1,0])/D
    a_31 = (L[1,0]*L[2,1]-L[1,1]*L[2,0])/D
    a_32 = -(L[0,0]*L[2,1]-L[0,1]*L[2,0])/D
    a_33 = (L[0,0]*L[1,1]-L[0,1]*L[1,0])/D
    
    #Construcció de vectors per la solució matricial del problema
    A = np.array([A_1,-1,A_3])
    B = np.array([B_1,0,B_3])
    X = np.array([R[0,0],R[0,1],R[0,2]])
    Y = np.array([R[1,0],R[1,1],R[1,2]])
    Z = np.array([R[2,0],R[2,1],R[2,2]])

    A_2 = -(a_21*np.dot(A,X)+a_22*np.dot(A,Y)+a_23*np.dot(A,Z))
    B_2 = -(a_21*np.dot(B,X)+a_22*np.dot(B,Y)+a_23*np.dot(B,Z))
    C_psi = -2*(X[1]*L[0,1]+Y[1]*L[1,1]+Z[1]*L[2,1])
    
    #Mòdul del vector posició de l'observador en la segona obs.
    R_2 = X[1]**2 + Y[1]**2 + Z[1]**2
    
    #Coeficients de l'equació de vuitè grau a resoldre
    a = -(C_psi*A_2 + A_2**2 + R_2**2)
    b = -mu*(C_psi*B_2 + 2*A_2*B_2)
    c = -mu**2*B_2**2
    
    #Solució numèrica de l'equació
    coeff = [1,0,a,0,0,b,0,0,c]
    sol = np.roots(coeff)
    res = ([])
    for i in range(len(sol)):
        if sol[i].real>0 and sol[i].imag==0:
            res = np.append(res,sol[i].real)
    if len(res)>=1:
        sol = res.min()
    
    u2 = mu/(sol**3)
    
    #Càlcul dels tres mòduls de posició geocèntrica de l'objecte
    D_1 = A_1 + B_1*u2
    D_3 = A_3 + B_3*u2
    A_1_estr = (a_11*np.dot(A,X) + a_12*np.dot(A,Y) + a_13*np.dot(A,Z))
    B_1_estr = (a_11*np.dot(B,X) + a_12*np.dot(B,Y) + a_13*np.dot(B,Z))
    A_3_estr = (a_31*np.dot(A,X) + a_32*np.dot(A,Y) + a_33*np.dot(A,Z))
    B_3_estr = (a_31*np.dot(B,X) + a_32*np.dot(B,Y) + a_33*np.dot(B,Z))

    rho_1 = (A_1_estr + B_1_estr*u2)/D_1
    rho_2 = A_2 + B_2*u2
    rho_3 = (A_3_estr + B_3_estr*u2)/D_3
    
    #Càlcul de posicions geocèntriques i consegüent loop de les fórmules de
    #Herrick-Gibbs per trobar la velocitat i millorar la precisió
    contador = 0
    while contador < 10000:
        rho = np.array([rho_1,rho_2,rho_3])
        
        #Vector de posició geocèntrica de l'objecte primera aprox.
        r = np.empty([3,3])
        for i in range(0,3):
            for j in range(0,3):
                r[i,j] = rho[j]*L[i,j] - R[i,j]
        
        #Fórmles de Herrick-Gibbs
        r_1 = np.array([r[0,0],r[1,0],r[2,0]])
        r_2 = np.array([r[0,1],r[1,1],r[2,1]])
        r_3 = np.array([r[0,2],r[1,2],r[2,2]])
        r1 = np.sqrt(np.dot(r_1,r_1))
        r2 = math.sqrt(np.dot(r_2,r_2))
        r3 = np.sqrt(np.dot(r_3,r_3))
        d1 = tau_3*(mu/(12*r1**3)-1/(tau_1*tau_13))
        d2 = (tau_1+tau_3)*(mu/(12*r2**3)-1/(tau_1*tau_3))
        d3 = -tau_1*(mu/(12*r3**3)+1/(tau_3*tau_13))
        
        rpunt_2 = np.empty([3])
        for i in range(0,3):
            rpunt_2[i] = -d1*r[i,0] + d2*r[i,1] + d3*r[i,2]
        
        V_2 = math.sqrt(np.dot(rpunt_2,rpunt_2))
        a = (2/r2-(V_2**2)/mu)**(-1)
        
        f1 = f_series(r2,V_2,tau_1)
        f3 = f_series(r2,V_2,tau_3)
        g1 = g_series(r2,V_2,tau_1)
        g3 = g_series(r2,V_2,tau_3)
        D_estr = f1*g3 - f3*g1
        c1 = g3/D_estr
        c2 = -1.0
        c3 = -g1/D_estr
        
        G = np.empty([3])
        for i in range(0,3):
            G[i]=c1*R[i,0]+c2*R[i,1]+c3*R[i,2]
        
        rho_1_n = (a_11*G[0]+a_12*G[1]+a_13*G[2])/c1
        rho_2_n = -(a_21*G[0]+a_22*G[1]+a_23*G[2])
        rho_3_n = (a_31*G[0]+a_32*G[1]+a_33*G[2])/c3
        
        #Toleràncies sobre les quals avaluar la precisió de les iteracions de
        #Herrick-Gibbs per millorar la precisió dels resultats
        tol1 = 10e-300
        tol2 = 10e-300
        tol3 = 10e-300
        #Aquesta millora ha servit per variar el resultat?
        if (abs(rho_1_n-rho_1)) < tol1 and (abs(rho_2_n-rho_2)) < tol2 and (abs(rho_3_n-rho_3)) < tol3:
            contador = 10000
        else:
            contador += 1
        
        rho_1 = rho_1_n
        rho_2 = rho_2_n
        rho_3 = rho_3_n

    rho = np.array([rho_1,rho_2,rho_3])
    
    #Vectors de posició geocèntrica de l'objecte amb la millora de Herrick-Gibbs
    for i in range(0,3):
        for j in range(0,3):
            r[i,j] = rho[j]*L[i,j] - R[i,j]
    r_2 = np.array([r[0,1],r[1,1],r[2,1]])
    
    #Vector velocitat de l'objecte al punt mig (segona obs.)
    rpunt_2 = np.empty([3])
    for i in range(0,3):
        rpunt_2[i] = -d1*r[i,0]+d2*r[i,1]+d3*r[i,2]
    
    
    
    #Resultat en metres:
    return r_2*6378136.6,rpunt_2*6378136.6

def Laplace(t,alpha,delta,phi,lambda_E,H):
    '''
    Parameters
    ----------
    t : Tres instants d'observació en segons
    alpha : Tres ascensions rectes topocèntriques d'observació en hores
    delta : Tres declinacions topocèntriques d'observació en graus
    phi : Tres latituds geodètiques en graus dels tres observadors
    lambda_E : Tres longituds est en graus dels tres observadors
    H : Tres altituds dels observadors en metres

    Returns
    -------
    r_2 : Posició en metres del segon instant d'observació
    rpunt_2 : Velocitat en metres per segon del segon instant d'observació
    '''
    alpha = alpha*math.pi/12
    delta = delta*math.pi/180
    lambda_E = lambda_E*math.pi/180
    phi = phi*math.pi/180
    # Unitats reduïdes (masses i radis equatorials terrestres)
    #Constants
    mu = 1
    k = 0.07436574/60

    #CÀLCUL
    #Variables temporals modificades
    tau1 = k*(t[0] - t[1])
    tau3 = k*(t[2] - t[1])
    s1 = -tau3/(tau1*(tau1 - tau3))
    s2 = -(tau3 + tau1)/(tau1*tau3)
    s3 = -tau1/(tau3*(tau3 - tau1))
    s4 = 2/(tau1*(tau1 - tau3))
    s5 = 2/(tau1*tau3)
    s6 = 2/(tau3*(tau3 - tau1))
    
    #Vectors normalitzats de coordenades topocèntriques de l'objecte
    L = np.empty([3,3])
    for i in range(0,3):
        L[0,i] = math.cos(delta[i])*math.cos(alpha[i])
        L[1,i] = math.cos(delta[i])*math.sin(alpha[i])
        L[2,i] = math.sin(delta[i])
    
    #Derivades primera i segona de L
    L1 = np.empty([3])
    L2 = np.empty([3])
    for i in range(0,3):
        L1[i] = s1*L[i,0] + s2*L[i,1] + s3*L[i,2]
        L2[i] = s4*L[i,0] + s5*L[i,1] + s6*L[i,2]
    
    #Poisció de l'observador
    R = np.empty([3,3])
    theta = np.empty([3])
    for i in range(0,3):
        theta[i] = sideraltime(t[1],lambda_E[i],t[i])
        R[0,i],R[1,i],R[2,i]=observador(phi[i],theta[i],H[i])
        R[0,i],R[1,i],R[2,i]=-R[0,i]/6378136.6,-R[1,i]/6378136.6,-R[2,i]/6378136.6
    
    #Derivades primera i segona de la posició de l'observador
    R1 = np.empty([3])
    R2 = np.empty([3])
    for i in range(0,3):
        R1[i] = s1*R[i,0] + s2*R[i,1] + s3*R[i,2]
        R2[i] = s4*R[i,0] + s5*R[i,1] + s6*R[i,2]
    
    #Determinants
    Delta = (2*((L[0,1]*L1[1]*L2[2] + L[1,1]*L1[2]*L2[0] + L[2,1]*L1[0]*L2[1]) 
             - (L[2,1]*L1[1]*L2[0] + L[0,1]*L1[2]*L2[1] + L[1,1]*L1[0]*L2[2])))
    Da = ((L[0,1]*L1[1]*R2[2] + L[1,1]*L1[2]*R2[0] + L[2,1]*L1[0]*R2[1]) 
             - (L[2,1]*L1[1]*R2[0] + L[0,1]*L1[2]*R2[1] + L[1,1]*L1[0]*R2[2]))
    Db = ((L[0,1]*L1[1]*R[2,1] + L[1,1]*L1[2]*R[0,1] + L[2,1]*L1[0]*R[1,1]) 
             - (L[2,1]*L1[1]*R[0,1] + L[0,1]*L1[2]*R[1,1] + L[1,1]*L1[0]*R[2,1]))
    Dc = ((L[0,1]*R1[1]*L2[2] + L[1,1]*R1[2]*L2[0] + L[2,1]*R1[0]*L2[1]) 
             - (L[2,1]*R1[1]*L2[0] + L[0,1]*R1[2]*L2[1] + L[1,1]*R1[0]*L2[2]))
    Dd = ((L[0,1]*R[1,1]*R[2,1] + L[1,1]*R[2,1]*R[0,1] + L[2,1]*R[0,1]*R[1,1]) 
             - (L[2,1]*R[1,1]*R[0,1] + L[0,1]*R[2,1]*R[1,1] + L[1,1]*R[0,1]*R[2,1]))
    
    A2 = 2*Da/Delta
    B2 = 2*Db/Delta
    C2 = Dc/Delta
    D2 = Dd/Delta
    C_phi = -2*(L[0,1]*R[0,1] + L[1,1]*R[1,1] + L[2,1]*R[2,1])
    
    # Mòdul del vector posició de l'observador en el segon instant de temps
    R_2 = R[0,1]**2 + R[1,1]**2 + R[2,1]**2
    
    #Coeficients de l'equació de vuitè grau a resoldre
    a = -(C_phi*A2 + A2**2 + R_2**2)
    b = -mu*(C_phi*B2 + 2*A2*B2)
    c = -mu**2*B2**2
    
    #Solució numèrica de l'equació
    coeff = [1,0,a,0,0,b,0,0,c]
    sol = np.roots(coeff)
    res = ([])
    for i in range(len(sol)):
        if sol[i].real>0 and sol[i].imag==0:
            res = np.append(res,sol[i].real)
    if len(res)>=1:
        sol = res.min()        
        
    rho2 = A2 + (mu*B2)/(sol**3)
    rho12 = C2 + (mu*D2)/(sol**3)
    
    #Vector posició i velocitat en la segona observació
    r = np.empty([3])
    r1 = np.empty([3])
    for i in range(0,3):
        r[i] = rho2*L[i,1] - R[i,1]
        r1[i] = rho12*L[i,1] + rho2*L1[i] - R1[i]
    
    #Resultat en metres:
    return r*6378136.6,r1*6378136.6

def prediccio(perc,semi):
    '''
    Parameters
    ----------
    perc : Percentatge de l'òrbita sobre el qual es faran observacions
    semi : array de dos valors: mínim i màxim valors possibles
            per la generació aleatòria del semieix major

    '''
    #Funció que dibuixa l'òrbita en 3D
    def plot(x,y,z):
        radi = 6378136.6
        def set_axes_equal(ax: plt.Axes):
            """Set 3D plot axes to equal scale.
        
            Make axes of 3D plot have equal scale so that spheres appear as
            spheres and cubes as cubes.  Required since `ax.axis('equal')`
            and `ax.set_aspect('equal')` don't work on 3D.
            """
            limits = np.array([
                ax.get_xlim3d(),
                ax.get_ylim3d(),
                ax.get_zlim3d(),
            ])
            origin = np.mean(limits, axis=1)
            radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
            _set_axes_radius(ax, origin, radius)
        def _set_axes_radius(ax, origin, radius):
            x, y, z = origin
            ax.set_xlim3d([x - radius, x + radius])
            ax.set_ylim3d([y - radius, y + radius])
            ax.set_zlim3d([z - radius, z + radius])
        plt.style.use('default')
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(x, y, z,'red')
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = radi*np.cos(u)*np.sin(v)
        y = radi*np.sin(u)*np.sin(v)
        z = radi*np.cos(v)
        ax.plot_surface(x, y, z, color="blue")
        ax.set_box_aspect([1,1,1])
        set_axes_equal(ax)
        plt.show()

    #Generació aleatòria dels elements orbitals
    a = rnd.uniform(semi[0],semi[1])
    e = rnd.random()
    inc = 360*rnd.random()
    O = 360*rnd.random()
    w = 360*rnd.random()
    nu = 360*rnd.random()
    #Període de l'òrbita
    periode = math.sqrt((4*math.pi**2*a**3)/(3.9860044188e14))
    #Elements cartesians de l'òrbita:
    pos,vel = ElementsCartesians(a,e,inc,O,w,nu)
    x = pos[0]
    y = pos[1]
    z = pos[2]
    vx = vel[0]
    vy = vel[1]
    vz = vel[2]
    r = np.array([x,y,z,vx,vy,vz])

    #Càlcul d'una òrbita
    t_inicial = JD(datetime.now())
    x,y,z,t,impacte = EulerSimple(r,periode)
    
    #Existeix l'òrbita? Aquest condicional és aquí perquè prèviament
    #s'havia inicialitzat l'òrbita amb la generació aleatòria dels vectors
    #posició i velocitat i així a vegades la trajectòria era d'impacte.
    if impacte==False:
        #Dibuix de l'òrbita en 3D
        plot(x,y,z)
        
        #Nombre de posicions corresponents al percentatge de l'òrbita
        #sobre el qual es vulguin generar observacions aleatòries
        seleccio = int(100001*perc)
        inici = int(rnd.uniform(0,(100001 - seleccio)))
        
        #Selecció aleatòria de tres observacions
        #Aleatòriament es generen els índexs que seriviran per extreure
        #la posició de les arrays que representen la trajectòria
        #que ens proporciona la funció EulerSimple
        i1 = int(rnd.uniform(inici,seleccio))
        i2 = int(rnd.uniform(inici,seleccio))
        i3 = int(rnd.uniform(inici,seleccio))
        
        #S'ordenen els índexs per saber quina és la primera i última observació
        if i1 > i2:
            i1,i2 = i2,i1
        if i1 > i3:
            i1,i3 = i3,i1
        if i2 > i3:
            i2,i3 = i3,i2
        
        #En quin moment es fan les observacions? (en segons)
        t1,t2,t3 = t[i1],t[i2],t[i3]
        t_1 = t_inicial*86400 + t1
        t_2 = t_inicial*86400 + t2
        t_3 = t_inicial*86400 + t3
        t = np.array([t_1,t_2,t_3])
        
        #Posició de l'objecte en aquestes observacions?
        x1,x2,x3 = x[i1],x[i2],x[i3]
        y1,y2,y3 = y[i1],y[i2],y[i3]
        z1,z2,z3 = z[i1],z[i2],z[i3]
        r1 = ([x1,y1,z1])
        r2 = ([x2,y2,z2])
        r3 = ([x3,y3,z3])
        r_t = (r1,r2,r3)
        
        #Es generen aleatòriament tres llocs d'observació
        phi = np.empty((3))
        longitud = np.empty((3))
        theta = np.empty((3))
        H = np.empty((3))
        for i in range(0,3):
            #Es comprova que l'objecte estigui a més de 10º d'altura
            h = 0.0
            while h<=10:
                phi[i] = 90*rnd.uniform(-1,1)
                longitud[i] = 360*rnd.random()
                theta[i] = sideraltime(t[1],longitud[i],t[i])
                H[i] = 4000*rnd.random()
                xo,yo,zo = observador(phi[i],theta[i],H[i])
                x_1,y_1,z_1 = r_t[i][0],r_t[i][1],r_t[i][2]
                A,h = A_h(x_1,y_1,z_1,xo,yo,zo)
        
        #Canvi de sistema de referència:
        #Posició de l'objecte de sistema geocèntric a topocèntric
        x1 = x1 - xo
        x2 = x2 - xo
        x3 = x3 - xo
        y1 = y1 - yo
        y2 = y2 - yo
        y3 = y3 - yo
        z1 = z1 - zo
        z2 = z2 - zo
        z3 = z3 - zo
        
        #Càlcul de les coordenades angulars topocèntriques de l'objecte
        #en les tres observacions
        RA1,dec1 = RA_dec(x1,y1,z1)
        RA2,dec2 = RA_dec(x2,y2,z2)
        RA3,dec3 = RA_dec(x3,y3,z3)
        
        alpha = np.array([RA1,RA2,RA3])
        delta = np.array([dec1,dec2,dec3])
        
        #Determinació de l'òrbita pel mètode de Gauss
        r_g,rpunt_g = Gauss(t,alpha,delta,phi,theta,H)
        ag,eg,ig,Og,wg,nug = ElementsKeplerians(r_g,rpunt_g)
        
        #Determinació de l'òrbita pel mètode de Laplace
        r_l,rpunt_l = Laplace(t,alpha,delta,phi,theta,H)
        al,el,il,Ol,wl,nul = ElementsKeplerians(r_l,rpunt_l)
        
        print('Òrbita real: ', a,e,inc,O,w,nu)
        print('Òrbita Gauss: ', ag,eg,ig,Og,wg,nug)
        print('Òrbita Laplace: ', al,el,il,Ol,wl,nul)
        
        print('Posició 2 real: ',r2)
        print('Posició,v 2 Gauss: ',r_g,rpunt_g)
        print('Posició,v 2 Laplace: ',r_l,rpunt_l)
        
        print("RA-dec 2 real: ",RA_dec(r2[0],r2[1],r2[2]))
        print("RA-dec 2 Gauss: ",RA_dec(r_g[0],r_g[1],r_g[2]))
        print("RA-dec 2 Laplace: ",RA_dec(r_l[0],r_l[1],r_l[2]))

        #Càlcul de l'òrbita determinada durant un quart del període de
        #l'òrbita original
        r_gauss = np.array([r_g[0],r_g[1],r_g[2],rpunt_g[0],rpunt_g[1],rpunt_g[2]])
        r_laplace = np.array([r_l[0],r_l[1],r_l[2],rpunt_l[0],rpunt_l[1],rpunt_l[2]])
        xg,yg,zg,tg,impacteg = EulerSimple(r_gauss,periode)
        xl,yl,zl,tl,impactel = EulerSimple(r_laplace,periode)
        

        
        #òrbites en coordenades celestes
        RA = ([])
        dec = ([])
        RAg = ([])
        decg = ([])
        RAl = ([])
        decl = ([])
        for i in range(0,len(x)):
            RAi,deci = RA_dec(x[i],y[i],z[i])
            RA = np.append(RA,RAi)
            dec = np.append(dec,deci)
        for i in range(0,len(xg)):
            RAgi,decgi = RA_dec(xg[i],yg[i],zg[i])
            RAg = np.append(RAg,RAgi)
            decg = np.append(decg,decgi)
        for i in range(0,len(xl)):
            RAli,decli = RA_dec(xl[i],yl[i],zl[i])
            RAl = np.append(RAl,RAli)
            decl = np.append(decl,decli)
        
        #Dibuix en coord. celestes de l'òrbita real
        plt.figure()
        plt.style.use('default')
        plt.scatter(RA,dec)
        plt.ylabel('Declinació (º)')
        plt.xlabel('Ascensió recta (h)')
        plt.xlim([0,24])
        
        #Dibuix en coord. celestes de l'òrbita real i les calculades
        plt.figure()
        plt.style.use('default')
        plt.scatter(RA,dec,label='Orb. real')
        plt.ylabel('Declinació (º)')
        plt.xlabel('Ascensió recta (h)')
        plt.xlim([0,24])
        plt.scatter(RAg,decg,color='black',label='Orb. Gauss')
        plt.scatter(RAl,decl,color='grey',label='Orb. Laplace')
        plt.scatter(RAg[0],decg[0],color='yellow')
        plt.scatter(RAl[0],decl[0],color='yellow',label="Inici d'orb. calculades")
            #Observacions
        RA1,dec1 = RA_dec(r_t[0][0],r_t[0][1],r_t[0][2])
        RA2,dec2 = RA_dec(r_t[1][0],r_t[1][1],r_t[1][2])
        RA3,dec3 = RA_dec(r_t[2][0],r_t[2][1],r_t[2][2])
        plt.scatter(RA1,dec1,color='red',label='1a obs.')
        plt.scatter(RA3,dec3,color='blue',label='3a obs.')
        plt.scatter(RA2,dec2,color='green',label='2a obs.')
        '''
        #Predicció al cap de cert temps, només si he usat el mateix període
        i4 = int(rnd.uniform(inici,seleccio) + 100001*perc*0.25)
        if i4>100001:
            i4 = i4-100001
        plt.scatter(RA[i4],dec[i4],color='orange',lw=5)
        plt.scatter(RAg[i4],decg[i4],color='pink',lw=3)
        plt.scatter(RAl[i4],decl[i4],color='brown',lw=1)
        '''
        plt.legend()
        
        
    else:
        prediccio(perc,semi)


prediccio(0.2,(1e7,1e8))
