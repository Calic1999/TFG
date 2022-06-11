import matplotlib.pyplot as plt
import numpy as np

#Solcuió de les equacions newtonianes de la gravetat 2D per Runge-Kutta-4
#Constants
G = 6.6843E-11
m1 = 5.97E24
m2 = 7.35E22
radi = 6378000
#Condicions inicials
x0 = 100000000
y0 = 100000000
vx0 = 100
vy0 = 0

passos = 10000
temps = 2592000
deltat = int(temps/passos)

x = ([x0])
y = ([y0])

def atraccio(r,x,y):
    G = 6.6843E-11
    m2 = 7.35E22
    return -G*m2*r/(np.sqrt((x**2+y**2))**3)

k = np.empty([4,4])


for i in range(0,passos):
    
    k[0,0] = vx0
    k[0,1] = vy0
    k[0,2] = atraccio(x0,x0,y0)
    k[0,3] = atraccio(y0,x0,y0)

    k[1,0] = vx0
    k[1,1] = vy0
    k[1,2] = atraccio(x0+deltat*k[0,2]/2,x0+deltat*k[0,2]/2,y0+deltat*k[0,3]/2)
    k[1,3] = atraccio(y0+deltat*k[0,3]/2,x0+deltat*k[0,3]/2,y0+deltat*k[0,3]/2)
    
    k[2,0] = vx0
    k[2,1] = vy0
    k[2,2] = atraccio(x0+deltat*k[1,2]/2,x0+deltat*k[1,2]/2,y0+deltat*k[1,3]/2)
    k[2,3] = atraccio(y0+deltat*k[1,3]/2,x0+deltat*k[1,3]/2,y0+deltat*k[1,3]/2)
    
    k[3,0] = vx0
    k[3,1] = vy0
    k[3,2] = atraccio(x0+deltat*k[2,2],x0+deltat*k[2,2],y0+deltat*k[2,3])
    k[3,3] = atraccio(y0+deltat*k[2,3],x0+deltat*k[2,3],y0+deltat*k[2,3])
    
    x1 = x0 + deltat*(k[0,0]+2*k[1,0]+2*k[2,0]+k[3,0])/6    
    y1 = y0 + deltat*(k[0,1]+2*k[1,1]+2*k[2,1]+k[3,1])/6
    vx1 = vx0 + deltat*(k[0,2]+2*k[1,2]+2*k[2,2]+k[3,2])/6
    vy1 = vy0 + deltat*(k[0,3]+2*k[1,3]+2*k[2,3]+k[3,3])/6    
    
  
    x = np.append(x,x1)
    y = np.append(y,y1)
    
    x0 = x1
    y0 = y1
    vx0 = vx1
    vy0 = vy1
    
    if (np.sqrt(x[-1]**2+y[-1]**2))<radi:
        print('Impacte!')
        break

plt.style.use('dark_background')
fig, ax = plt.subplots()
plt.plot(x, y, linewidth=2.0,c='r')
theta = np.linspace(0,2*3.1415,1000000)
plt.fill_between(radi*np.cos(theta),radi*np.sin(theta),color='b')
plt.axis('off')
plt.axis('equal')
plt.show()

