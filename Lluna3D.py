import matplotlib.pyplot as plt
import numpy as np

#Solcuió de les equacions newtonianes de la gravetat 2D per Euler
#Constants
G = 6.6843E-11
m1 = 5.97E24
m2 = 7.35E22
radi = 6378000
#Condicions inicials
x0 = 80000000
y0 = 100000000
z0 = 100000000
vx0 = 1000
vy0 = 0
vz0 = 50

passos = 10000
temps = 2592000
deltat = int(temps/passos)

x = ([x0])
y = ([y0])
z = ([z0])

for i in range(0,passos):
    if i==0:
        x1 = x0 + deltat*vx0
        y1 = y0 + deltat*vy0
        z1 = z0 + deltat*vz0
        vx1 = vx0 - deltat*G*m1*x0/(np.sqrt((x0**2+y0**2+z0**2))**3)
        vy1 = vy0 - deltat*G*m1*y0/(np.sqrt((x0**2+y0**2+z0**2))**3)
        vz1 = vz0 - deltat*G*m1*z0/(np.sqrt(x0**2+y0**2+z0**2)**3)
    
    x2 = x0 + 2*deltat*vx1
    y2 = y0 + 2*deltat*vy1
    z2 = z0 + 2*deltat*vz1
    vx2 = vx0 - 2*deltat*G*m1*x1/(np.sqrt((x1**2+y1**2+z1**2))**3)
    vy2 = vy0 - 2*deltat*G*m1*y1/(np.sqrt((x1**2+y1**2+z1**2))**3)
    vz2 = vz0 - 2*deltat*G*m1*z1/(np.sqrt((x1**2+y1**2+z1**2))**3)
    
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
    
    if (np.sqrt(x[-1]**2+y[-1]**2+z[-1]**2))<radi:
        print('Impacte!')
        break

fig = plt.figure()
plt.style.use('dark_background')
ax = plt.axes(projection='3d')
ax.plot3D(x, y, z,'red')
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = radi*np.cos(u)*np.sin(v)
y = radi*np.sin(u)*np.sin(v)
z = radi*np.cos(v)
ax.plot_surface(x, y, z, color="blue")
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
ax.axes.zaxis.set_ticklabels([])
plt.show()

