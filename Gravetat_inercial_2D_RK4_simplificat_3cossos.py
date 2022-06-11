import matplotlib.pyplot as plt
import numpy as np

#Solcuió de les equacions newtonianes de la gravetat 2D per Runge-Kutta-4
#Constants
G = 6.6843E-11
m1 = 1E22
m2 = 1E22
m3 = 1E22
radi = 6378000
vel_max = 299792458*0.001
aprop_max = 1000

#Condicions inicials
x1_0 = 100000
y1_0 = 100000
vx1_0 = 100
vy1_0 = 0

x2_0 = 0
y2_0 = 0
vx2_0 = 0
vy2_0 = 100

x3_0 = 0
y3_0 = 100000
vx3_0 = 100
vy3_0 = 0


x1 = ([x1_0])
y1 = ([y1_0])
x2 = ([x2_0])
y2 = ([y2_0])
x3 = ([x3_0])
y3 = ([y3_0])


def atraccio(x,y,d,m):
    G = 6.6843E-11
    return -G*m*(x-y)/(d**3)

def dist(x1,y1,x2,y2):
    d = np.sqrt((x2-x1)**2+(y2-y1)**2)
    return d


#RUNGE-KUTTA 4

passos = 6000
temps = 3600
h = 0.01

k = np.empty([4,12])

for i in range(0,passos):
    
    d12 = dist(x1_0,y1_0,x2_0,y2_0)
    d13 = dist(x1_0,y1_0,x3_0,y3_0)
    d23 = dist(x2_0,y2_0,x3_0,y3_0)
    k[0,0] = vx1_0
    k[0,1] = vy1_0
    k[0,2] = atraccio(x1_0,x2_0,d12,m2)+atraccio(x1_0,x3_0,d13,m3)
    k[0,3] = atraccio(y1_0,y2_0,d12,m2)+atraccio(y1_0,y3_0,d13,m3)
    k[0,4] = vx2_0
    k[0,5] = vy2_0
    k[0,6] = atraccio(x2_0,x1_0,d12,m1)+atraccio(x2_0,x3_0,d23,m3)
    k[0,7] = atraccio(y2_0,y1_0,d12,m1)+atraccio(y2_0,y3_0,d23,m3)
    k[0,8] = vx3_0
    k[0,9] = vy3_0
    k[0,10] = atraccio(x3_0,x1_0,d13,m1)+atraccio(x3_0,x2_0,d23,m2)
    k[0,11] = atraccio(y3_0,y1_0,d13,m1)+atraccio(y3_0,y2_0,d23,m2)

    d12 = dist(x1_0+h*k[0,0]/2,y1_0+h*k[0,1]/2,x2_0+h*k[0,4]/2,y2_0+h*k[0,5]/2)
    d13 = dist(x1_0+h*k[0,0]/2,y1_0+h*k[0,1]/2,x3_0+h*k[0,8]/2,y3_0+h*k[0,9]/2)
    d23 = dist(x2_0+h*k[0,4]/2,y2_0+h*k[0,5]/2,x3_0+h*k[0,8]/2,y3_0+h*k[0,9]/2)
    k[1,0] = vx1_0+h*k[0,2]/2
    k[1,1] = vy1_0+h*k[0,3]/2
    k[1,2] = atraccio(x1_0+h*k[0,0]/2,x2_0+h*k[0,4]/2,d12,m2)+atraccio(x1_0+h*k[0,0]/2,x3_0+h*k[0,8]/2,d13,m3)
    k[1,3] = atraccio(y1_0+h*k[0,1]/2,y2_0+h*k[0,5]/2,d12,m2)+atraccio(y1_0+h*k[0,1]/2,y3_0+h*k[0,9]/2,d13,m3)
    k[1,4] = vx2_0+h*k[0,6]/2
    k[1,5] = vy2_0+h*k[0,7]/2
    k[1,6] = atraccio(x2_0+h*k[0,4]/2,x1_0+h*k[0,0]/2,d12,m1)+atraccio(x2_0+h*k[0,4]/2,x3_0+h*k[0,8]/2,d23,m3)
    k[1,7] = atraccio(y2_0+h*k[0,5]/2,y1_0+h*k[0,1]/2,d12,m1)+atraccio(y2_0+h*k[0,5]/2,y3_0+h*k[0,9]/2,d23,m3)
    k[1,8] = vx3_0+h*k[0,10]/2
    k[1,9] = vy3_0+h*k[0,11]/2
    k[1,10] = atraccio(x3_0+h*k[0,8]/2,x1_0+h*k[0,0]/2,d13,m1)+atraccio(x3_0+h*k[0,8]/2,x2_0+h*k[0,4]/2,d23,m2)
    k[1,11] = atraccio(y3_0+h*k[0,9]/2,y1_0+h*k[0,1]/2,d13,m1)+atraccio(y3_0+h*k[0,9]/2,y2_0+h*k[0,5]/2,d23,m2)

    d12 = dist(x1_0+h*k[1,0]/2,y1_0+h*k[1,1]/2,x2_0+h*k[1,4]/2,y2_0+h*k[1,5]/2)
    d13 = dist(x1_0+h*k[1,0]/2,y1_0+h*k[1,1]/2,x3_0+h*k[1,8]/2,y3_0+h*k[1,9]/2)
    d23 = dist(x2_0+h*k[1,4]/2,y2_0+h*k[1,5]/2,x3_0+h*k[1,8]/2,y3_0+h*k[1,9]/2)
    k[2,0] = vx1_0+h*k[1,2]/2
    k[2,1] = vy1_0+h*k[1,3]/2
    k[2,2] = atraccio(x1_0+h*k[1,0]/2,x2_0+h*k[1,4]/2,d12,m2)+atraccio(x1_0+h*k[1,0]/2,x3_0+h*k[1,8]/2,d13,m3)
    k[2,3] = atraccio(y1_0+h*k[1,1]/2,y2_0+h*k[1,5]/2,d12,m2)+atraccio(y1_0+h*k[1,1]/2,y3_0+h*k[1,9]/2,d13,m3)
    k[2,4] = vx2_0+h*k[1,6]/2
    k[2,5] = vy2_0+h*k[1,7]/2
    k[2,6] = atraccio(x2_0+h*k[1,4]/2,x1_0+h*k[1,0]/2,d12,m1)+atraccio(x2_0+h*k[1,4]/2,x3_0+h*k[1,8]/2,d23,m3)
    k[2,7] = atraccio(y2_0+h*k[1,5]/2,y1_0+h*k[1,1]/2,d12,m1)+atraccio(y2_0+h*k[1,5]/2,y3_0+h*k[1,9]/2,d23,m3)
    k[2,8] = vx3_0+h*k[1,10]/2
    k[2,9] = vy3_0+h*k[1,11]/2
    k[2,10] = atraccio(x3_0+h*k[1,8]/2,x1_0+h*k[1,0]/2,d13,m1)+atraccio(x3_0+h*k[1,8]/2,x2_0+h*k[1,4]/2,d23,m2)
    k[2,11] = atraccio(y3_0+h*k[1,9]/2,y1_0+h*k[1,1]/2,d13,m1)+atraccio(y3_0+h*k[1,9]/2,y2_0+h*k[1,5]/2,d23,m2)

    d12 = dist(x1_0+h*k[2,0],y1_0+h*k[2,1],x2_0+h*k[2,4],y2_0+h*k[2,5])
    d13 = dist(x1_0+h*k[2,0],y1_0+h*k[2,1],x3_0+h*k[2,8],y3_0+h*k[2,9])
    d23 = dist(x2_0+h*k[2,4],y2_0+h*k[2,5],x3_0+h*k[2,8],y3_0+h*k[2,9])
    k[3,0] = vx1_0+h*k[2,2]
    k[3,1] = vy1_0+h*k[2,3]
    k[3,2] = atraccio(x1_0+h*k[2,0],x2_0+h*k[2,4],d12,m2)+atraccio(x1_0+h*k[2,0],x3_0+h*k[2,8],d13,m3)
    k[3,3] = atraccio(y1_0+h*k[2,1],y2_0+h*k[2,5],d12,m2)+atraccio(y1_0+h*k[2,1],y3_0+h*k[2,9],d13,m3)
    k[3,4] = vx2_0+h*k[2,6]
    k[3,5] = vy2_0+h*k[2,7]
    k[3,6] = atraccio(x2_0+h*k[2,4],x1_0+h*k[2,0],d12,m1)+atraccio(x2_0+h*k[2,4],x3_0+h*k[2,8],d23,m3)
    k[3,7] = atraccio(y2_0+h*k[2,5],y1_0+h*k[2,1],d12,m1)+atraccio(y2_0+h*k[2,5],y3_0+h*k[2,9],d23,m3)
    k[3,8] = vx3_0+h*k[2,10]
    k[3,9] = vy3_0+h*k[2,11]
    k[3,10] = atraccio(x3_0+h*k[2,8],x1_0+h*k[2,0],d13,m1)+atraccio(x3_0+h*k[2,8],x2_0+h*k[2,4],d23,m2)
    k[3,11] = atraccio(y3_0+h*k[2,9],y1_0+h*k[2,1],d13,m1)+atraccio(y3_0+h*k[2,9],y2_0+h*k[2,5],d23,m2)

    x1_1 = x1_0 + h*(k[0,0]+2*k[1,0]+2*k[2,0]+k[3,0])/6    
    y1_1 = y1_0 + h*(k[0,1]+2*k[1,1]+2*k[2,1]+k[3,1])/6
    vx1_1 = vx1_0 + h*(k[0,2]+2*k[1,2]+2*k[2,2]+k[3,2])/6
    vy1_1 = vy1_0 + h*(k[0,3]+2*k[1,3]+2*k[2,3]+k[3,3])/6    
    x2_1 = x2_0 + h*(k[0,4]+2*k[1,4]+2*k[2,4]+k[3,4])/6    
    y2_1 = y2_0 + h*(k[0,5]+2*k[1,5]+2*k[2,5]+k[3,5])/6
    vx2_1 = vx2_0 + h*(k[0,6]+2*k[1,6]+2*k[2,6]+k[3,6])/6
    vy2_1 = vy2_0 + h*(k[0,7]+2*k[1,7]+2*k[2,7]+k[3,7])/6
    x3_1 = x3_0 + h*(k[0,8]+2*k[1,8]+2*k[2,8]+k[3,8])/6    
    y3_1 = y3_0 + h*(k[0,9]+2*k[1,9]+2*k[2,9]+k[3,9])/6
    vx3_1 = vx3_0 + h*(k[0,10]+2*k[1,10]+2*k[2,10]+k[3,10])/6
    vy3_1 = vy3_0 + h*(k[0,11]+2*k[1,11]+2*k[2,11]+k[3,11])/6 


    x1 = np.append(x1,x1_1)
    y1 = np.append(y1,y1_1)
    x2 = np.append(x2,x2_1)
    y2 = np.append(y2,y2_1)
    x3 = np.append(x3,x3_1)
    y3 = np.append(y3,y3_1)
    
    x1_0 = x1_1
    y1_0 = y1_1
    vx1_0 = vx1_1
    vy1_0 = vy1_1
    x2_0 = x2_1
    y2_0 = y2_1
    vx2_0 = vx2_1
    vy2_0 = vy2_1
    x3_0 = x3_1
    y3_0 = y3_1
    vx3_0 = vx3_1
    vy3_0 = vy3_1


    if (dist(x1_0,y1_0,x2_0,y2_0))<aprop_max or (dist(x1_0,y1_0,x3_0,y3_0))<aprop_max or (dist(x2_0,y2_0,x3_0,y3_0))<aprop_max:
        if (dist(x1_0,y1_0,x2_0,y2_0))<100:
            print('Impacte 1 2!')
        if (dist(x1_0,y1_0,x3_0,y3_0))<100:
            print('Impacte 1 3!')
        if (dist(x3_0,y3_0,x2_0,y2_0))<100:
            print('Impacte 3 2!')
        break
    if (np.sqrt((vx1_1)**2+(vy1_1)**2))>vel_max or (np.sqrt((vx2_1)**2+(vy2_1)**2))>vel_max or (np.sqrt((vx3_1)**2+(vy3_1)**2))>vel_max:
        print('Frena fitipaldi!')
        break
    
    if i%60==0:
        plt.style.use('dark_background')
        fig, ax = plt.subplots()
        plt.plot(x1, y1, linewidth=2.0,c='r')
        plt.plot(x2, y2, linewidth=2.0,c='b')
        plt.plot(x3, y3, linewidth=2.0,c='y')
        plt.axis('off')
        plt.axis('equal')
        plt.show()

plt.plot(x1, y1, linewidth=2.0,c='r')
plt.plot(x2, y2, linewidth=2.0,c='b')
plt.plot(x3, y3, linewidth=2.0,c='y')
plt.axis('off')
plt.axis('equal')
plt.show()



