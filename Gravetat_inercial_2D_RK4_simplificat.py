import matplotlib.pyplot as plt
import numpy as np

#Solcuió de les equacions newtonianes de la gravetat 2D per Runge-Kutta-4
#Constants
G = 6.6843E-11
m1 = 6E24
m2 = 7E22
radi1 = 6378000
radi2 = 1737000
col = radi1+radi1

#Condicions inicials
x1_0 = 0
y1_0 = 0
vx1_0 = 0
vy1_0 = 0

x2_0 = 384400000
y2_0 = 0
vx2_0 = 0
vy2_0 = np.sqrt(G*m1/x2_0)

x1 = ([x1_0])
y1 = ([y1_0])
x2 = ([x2_0])
y2 = ([y2_0])

def atraccio(x,y,d,m):
    G = 6.6843E-11
    return -G*m*(x-y)/(d**3)


def dist(x1,y1,x2,y2):
    return np.sqrt((x2-x1)**2+(y2-y1)**2)

#RUNGE-KUTTA 4

passos = 10000
temps = 25920000
h = int(temps/passos)

k = np.empty([4,8])

for i in range(0,passos):
    
    d = dist(x1_0,y1_0,x2_0,y2_0)
    k[0,0] = vx1_0
    k[0,1] = vy1_0
    k[0,2] = atraccio(x1_0,x2_0,d,m2)
    k[0,3] = atraccio(y1_0,y2_0,d,m2)
    k[0,4] = vx2_0
    k[0,5] = vy2_0
    k[0,6] = atraccio(x2_0,x1_0,d,m1)
    k[0,7] = atraccio(y2_0,y1_0,d,m1)
    
    d = dist(x1_0+h*k[0,0]/2,y1_0+h*k[0,1]/2,x2_0+h*k[0,4]/2,y2_0+h*k[0,5]/2)
    k[1,0] = vx1_0+h*k[0,2]/2
    k[1,1] = vy1_0+h*k[0,3]/2
    k[1,2] = atraccio(x1_0+h*k[0,0]/2,x2_0+h*k[0,4]/2,d,m2)
    k[1,3] = atraccio(y1_0+h*k[0,1]/2,y2_0+h*k[0,5]/2,d,m2)
    k[1,4] = vx2_0+h*k[0,6]/2
    k[1,5] = vy2_0+h*k[0,7]/2
    k[1,6] = atraccio(x2_0+h*k[0,4]/2,x1_0+h*k[0,0]/2,d,m1)
    k[1,7] = atraccio(y2_0+h*k[0,5]/2,y1_0+h*k[0,1]/2,d,m1)
    
    d = dist(x1_0+h*k[1,0]/2,y1_0+h*k[1,1]/2,x2_0+h*k[1,4]/2,y2_0+h*k[1,5]/2)
    k[2,0] = vx1_0+h*k[1,2]/2
    k[2,1] = vy1_0+h*k[1,3]/2
    k[2,2] = atraccio(x1_0+h*k[1,0]/2,x2_0+h*k[1,4]/2,d,m2)
    k[2,3] = atraccio(y1_0+h*k[1,1]/2,y2_0+h*k[1,5]/2,d,m2)
    k[2,4] = vx2_0+h*k[1,6]/2
    k[2,5] = vy2_0+h*k[1,7]/2
    k[2,6] = atraccio(x2_0+h*k[1,4]/2,x1_0+h*k[1,0]/2,d,m1)
    k[2,7] = atraccio(y2_0+h*k[1,5]/2,y1_0+h*k[1,1]/2,d,m1)
    
    d = dist(x1_0+h*k[2,0],y1_0+h*k[2,1],x2_0+h*k[2,4],y2_0+h*k[2,5])
    k[3,0] = vx1_0+h*k[2,2]
    k[3,1] = vy1_0+h*k[2,3]
    k[3,2] = atraccio(x1_0+h*k[2,0],x2_0+h*k[2,4],d,m2)
    k[3,3] = atraccio(y1_0+h*k[2,1],y2_0+h*k[2,5],d,m2)
    k[3,4] = vx2_0+h*k[2,6]
    k[3,5] = vy2_0+h*k[2,7]
    k[3,6] = atraccio(x2_0+h*k[2,4],x1_0+h*k[2,0],d,m1)
    k[3,7] = atraccio(y2_0+h*k[2,5],y1_0+h*k[2,1],d,m1)

    
    x1_1 = x1_0 + h*(k[0,0]+2*k[1,0]+2*k[2,0]+k[3,0])/6    
    y1_1 = y1_0 + h*(k[0,1]+2*k[1,1]+2*k[2,1]+k[3,1])/6
    vx1_1 = vx1_0 + h*(k[0,2]+2*k[1,2]+2*k[2,2]+k[3,2])/6
    vy1_1 = vy1_0 + h*(k[0,3]+2*k[1,3]+2*k[2,3]+k[3,3])/6    
    x2_1 = x2_0 + h*(k[0,4]+2*k[1,4]+2*k[2,4]+k[3,4])/6    
    y2_1 = y2_0 + h*(k[0,5]+2*k[1,5]+2*k[2,5]+k[3,5])/6
    vx2_1 = vx2_0 + h*(k[0,6]+2*k[1,6]+2*k[2,6]+k[3,6])/6
    vy2_1 = vy2_0 + h*(k[0,7]+2*k[1,7]+2*k[2,7]+k[3,7])/6    
    
  
    x1 = np.append(x1,x1_1)
    y1 = np.append(y1,y1_1)
    x2 = np.append(x2,x2_1)
    y2 = np.append(y2,y2_1)
    


    x1_0 = x1_1
    y1_0 = y1_1
    vx1_0 = vx1_1
    vy1_0 = vy1_1
    x2_0 = x2_1
    y2_0 = y2_1
    vx2_0 = vx2_1
    vy2_0 = vy2_1
    
    if (dist(x1_0,y1_0,x2_0,y2_0))<col:
        print('Impacte!')
        break
    elif (np.sqrt((vx1_1)**2+(vy1_1)**2))>299792458 or (np.sqrt((vx2_1)**2+(vy2_1)**2))>299792458:
        print('Frena fitipaldi!')
        break
    
    if i%100==0:
        plt.style.use('dark_background')
        fig, ax = plt.subplots()
        plt.plot(x1, y1, linewidth=2.0,c='r')
        plt.plot(x2, y2, linewidth=2.0,c='b')
        plt.axis('off')
        plt.axis('equal')
        plt.show()

plt.style.use('dark_background')
fig, ax = plt.subplots()
plt.plot(x1, y1, linewidth=2.0,c='r')
plt.plot(x2, y2, linewidth=2.0,c='b')
plt.axis('off')
plt.axis('equal')
plt.show()

