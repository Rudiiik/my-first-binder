#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cmath
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from IPython.display import display
from ipywidgets import IntProgress
#from sympy import *
import cProfile


# In[2]:


#геометрия поверхности и ее дискретизация
N = 20
M = N
Lx = 0.001
hx = Lx/N
Ly = Lx
hy = hx
h = hx
L = Lx

E = np.zeros((N-1,M-1), dtype='complex') #массив полей в точке поверхности
k = np.zeros((N-1,M-1), dtype='float') #массив вв
Ex = np.zeros((N,M), dtype='complex') #массив производных по (x,y,z)
Ey = np.zeros((N,M), dtype='complex')
Ez = np.zeros((N,M), dtype='complex')
En = np.zeros((N-1,M-1), dtype='complex')
norm_ar = np.zeros((N-1,M-1, 3))

#r - вектор от точки (i,j) до Р
r = np.zeros((N-1, M-1, 3), dtype='float') 
r_ar = np.zeros((N-1,M-1), dtype='float')
R_ar = np.zeros((N-1,M-1), dtype='float')


# In[225]:


#поверхность Z(x,y) -> Z(i,j)
Z = np.ones((N,M))
Z.fill(0.002)



# In[226]:


#константы
p = 0.
w0 = 10*10**(-6)
lambd = 1.5e-6
k.fill(2*np.pi/lambd)
A0 = 1000.
zr = float(np.pi*(w0)**2/lambd)


# In[227]:


#функция ГП
def E_g(x, y, z):
    
    w_z = w0*np.sqrt(1+(z/zr)**2)
    R_z = z*(1 + (zr/z)**2)
    q_z = 1/(1/R_z + (1j*lambd)/(np.pi*w_z**2))
    
    Eg = ((A0*np.exp(1j*(p - (2*np.pi*z)/lambd - (1j*np.pi* (x**2 + y**2))/(np.pi*w0**2 + 1j*z *lambd)))) *np.pi)/np.sqrt(np.pi**2+(z**2*lambd**2)/w0**4)
    
    
    return Eg

def Eref(x, y, z):
    
    #w_z = w0*np.sqrt(1+(z/zr)**2)
    #Rref_z = (-z*(1+(zr/z)**2))
    #qref_z = 1/(1/Rref_z + (1j*lambd)/(np.pi*(w_z)**2))

    #Eg_ref = A0*(w0/w_z)*cmath.exp(1j*(p + (k/(2*qref_z))*(x**2 + y**2)) - 1j*k*z)
    Eg_ref = ((A0*np.exp(1j*(p - (2*np.pi*z)/lambd - (np.pi* (x**2 + y**2))/( 1j*np.pi*w0**2 + z *lambd))) )*np.pi)/np.sqrt(np.pi**2+(z**2*lambd**2)/w0**4)
    
    return Eg_ref
    
#функкции производных ГП 
#def dE_Gx(x, y, z, h):
#    
#    dEgx = (Eref(x + h, y, z) - Eref(x - h, y, z))/(2 * h)
#     
#    return dEgx
#
#def dE_Gy(x, y, z, h_y):
#    
#    dEgy = (Eref(x, y + h, z) - Eref(x, y - h_x, z))/(2 * h)
#    
#    return dEgy
#
#def dE_Gz(x, y, z, h):
#    
#    dEgz = (Eref(x, y, z + h) - Eref(x, y, z - h))/(2 * h)
#        
#    return dEgz
#


# In[228]:


def dE_gx(x, y, z):
    
    dEgx = -(2*1j* A0*np.exp( 1j* (p - (2 *np.pi* z)/lambd - (np.pi*(x**2 + y**2))/(1j*np.pi* w0**2 + z *lambd)))* np.pi**2*x)/((1j*np.pi * w0**2 + z *lambd)* np.sqrt(np.pi**2 + (z**2 *lambd**2)/w0**4))
             
    return dEgx

def dE_gy(x, y, z):
    
    dEgy = -(2*1j* A0*np.exp( 1j* (p - (2 *np.pi* z)/lambd - (np.pi *(x**2 + y**2))/
                (1j*np.pi* w0**2 + z *lambd)))* np.pi**2*y)/((1j*np.pi * w0**2 +  
    z *lambd)* np.sqrt(np.pi**2 + (z**2 *lambd**2)/w0**4))
    
    return dEgy

def dE_gz(x, y, z):
    
    dEgz = -((2*1j* A0*np.exp( 1j* (p - (2 *np.pi* z)/lambd - (np.pi *(x**2 + y**2))/
                (1j*np.pi* w0**2 + z *lambd)))* np.pi*z*lambd**2)/
             (w0**4*(np.pi**2 + (z**2*lambd**2)/w0**4)**(3/2))) + (1j* A0*np.exp(1j*(p - (2*np.pi*z)/lambd 
            - (np.pi*(x**2 + y**2))/(1j*np.pi* w0**2 + z*lambd)))*np.pi*(-((2*np.pi)/lambd) + (np.pi*(x**2 +  
       y**2)*lambd)/(1j*np.pi*w0**2 +  z*lambd)**2))*np.sqrt(np.pi**2 + (z**2*lambd**2)/w0**4)
        
    return dEgz


# In[261]:


#рассчет истинных координат
coord_array = np.zeros((N,M,3), dtype=np.float64)

for i in range(0, N):
    for j in range(0, M):
        coord_array[i][j][0] = (-L/2) + i*h
        coord_array[i][j][1] =  (-L/2) + j*h
        coord_array[i][j][2] = (-L/2) + i*h
        
d = coord_array*1.


# In[246]:


def Norm(i,j):
    d1 = coord_array*1.
    d2 = coord_array*1.
    n = coord_array*1.
    for i in range(0, N-1):
        d1 = coord_array[i+1][j] - coord_array[i][j]
        for j in range(0, M-1):
            d2 = coord_array[i][j+1] - coord_array[i][j]
    
            n[i][j][0] = d1[1]*d2[2] - d1[2]*d2[1] 
            n[i][j][1] = d1[2]*d2[0] - d1[0]*d2[2]
            n[i][j][2] = d1[0]*d2[1] - d1[1]*d2[0]  
    
    return n[i][j]/(np.sqrt(n[i][j][0]**2 + n[i][j][1]**2 + n[i][j][2]**2))

def E_n(x, y, z):
    
    En =  dE_gx(x, y, z)*Norm(x, y)[0] + dE_gy(x, y, z)*Norm(x, y)[1] + dE_gz(x, y, z)*Norm(x, y)[2]
   
    return En


# In[247]:


#рассчеты неизменяемых массивов
for i in range(0, N-1):
    for j in range(0, M-1):
        E[i][j] = Eref(coord_array[i][j][0],coord_array[i][j][1],coord_array[i][j][2])
        Ex[i][j] = dE_gx(coord_array[i][j][0],coord_array[i][j][1],coord_array[i][j][2])
        Ey[i][j] = dE_gy(coord_array[i][j][0],coord_array[i][j][1],coord_array[i][j][2])
        Ez[i][j] = dE_gz(coord_array[i][j][0],coord_array[i][j][1],coord_array[i][j][2])
        norm = (Norm(i, j))
        norm_ar[i][j] = norm
        En[i][j] = Ex[i][j]*norm_ar[i][j][0] + Ey[i][j]*norm_ar[i][j][1] + Ez[i][j]*norm_ar[i][j][2] 


# In[248]:


#рассчетные точки
n = 100
m = 100
lx = 0.005
ly = lx
hnx = lx/n
hn = hnx


# In[249]:


cal_ar = np.zeros((n,m,3), dtype=np.float64)

for u in range(0, n):
    for v in range(0, m):
        cal_ar[u][v][0] = -lx/2 + u*hn
        cal_ar[u][v][1] = -ly/2 + v*hn
        cal_ar[u][v][2] = 0.0005
        
      

 #задаю рассчетные точки         
P = cal_ar*1.
#P = [(1, 0, 0), (5, 0 , 0), (7, 0 , 0)]
d = coord_array*1.


# In[250]:


Pc = []
def P_c(r, i):
    vect = (-r*np.sin(i), 0, r*np.cos(i)+0.02)
    return vect
    


for i in range(0,90):
    Pc.append(P_c(0.01, (i*np.pi)/180))
len(Pc)


# In[251]:


#функция рассчета косинуса м нормалью и направлением
cos = np.zeros((N-1, M-1), dtype='float') 
def cos_a(norm, r, R):
    
    vector_1 = norm
    vector_2 = r
    #norm_1 = np.sqrt(vector_1[0]**2 + vector_1[1]**2 + vector_1[2]**2)
    #norm_2 = np.sqrt(vector_2[0]**2 + vector_2[1]**2 + vector_2[2]**2)
    scalar = vector_1[0] * vector_2[0] + vector_1[1] * vector_2[1] + vector_1[2] * vector_2[2]
    cos = scalar * R

    return cos


# In[252]:


#рассчет ИК для 1й рассчетной точки 
#a = cal_ar[u][v][0]
#b = cal_ar[u][v][1]
#c = cal_ar[u][v][2]
def KI(a, b, c):
    for i in range(0, N-1):
        #r[i] = P[u][v] - d[i][j]
        for j in range(0, M-1):
            r[i][j] = [a,b,c] - d[i][j]
            r_ar[i][j] = np.sqrt(r[i][j][0]**2+r[i][j][1]**2+r[i][j][2]**2)
            R_ar[i][j] = 1./r_ar[i][j]
            cos[i][j] = cos_a(norm_ar[i][j], r[i][j], R_ar[i][j])
    #print(1)
            
    S = h**2
    res = np.sum(((1/(4*np.pi))*((1j* k + R_ar)*E*cos-En)*np.exp(-1j*k*r_ar)*R_ar*S))
    #print(2)
    return res



# In[253]:


res_ar_c = np.zeros((n-1,m-1), dtype='complex')
Q_c = np.zeros((n-1,m-1), dtype='float')


# In[254]:


res_ar = np.zeros((90, 3), dtype='complex')
Q = np.zeros((90), dtype='float')


# In[255]:


#cProfile.run('KI(0.,0.,0.001)')


# In[256]:


get_ipython().run_cell_magic('time', '', '#f = IntProgress(min = 0, max = (n-1)*(n-1))\n#display(f)\nfor u in range(0, n-1):\n    for v in range(0, m-1):\n        res_ar[u][v] = KI(cal_ar[u][v][0],cal_ar[u][v][1],cal_ar[u][v][2])\n        #f.value += 1\n#Q_c = abs(res_ar)**2 \nlen(res_ar)\n')


# In[257]:


Xc = np.zeros((90), dtype='float')
for i in range(len(Pc)):
    res_ar[i] = KI(Pc[i][0], Pc[i][1], Pc[i][2])
    Xc[i] = 90 - i
Q = abs(res_ar)**2 
len(Xc)


# In[258]:


plt.imshow(Q)


# In[259]:


#plt.imshow(Q.T, origin='lower',interpolation='nearest',cmap='Blues', norm=mc.Normalize(vmin=0,vmax=Q.max()))

plt.plot(Xc, Q)


# In[260]:


df = pd.read_csv('Popytka_2_udachnaya_-_44.csv', sep = ';')
df.columns = ['X', 'Y']
df1 = df[0:27]
q = np.array(df1['X'])
w = np.array(df1['Y'])


# In[163]:


plt.plot(q, w/800)
plt.plot(Xc, Q/(4.4*10**7))


# In[ ]:





# In[ ]:




