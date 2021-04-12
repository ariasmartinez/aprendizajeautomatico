# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: Celia Arias Martínez
"""
import numpy as np
import matplotlib.pyplot as plt


# Fijamos la semilla
np.random.seed(1)


def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gaus(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0])) 
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
    
    return out


def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b



# EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida correspondiente

x_1 = simula_unif(50, 2, [-50,50])
#CODIGO DEL ESTUDIANTE

plt.scatter(x_1[:,0], x_1[:,1], c ='r') #pintamos dicha muestra
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

x_2 = simula_gaus(50, 2, np.array([5,7]))
#CODIGO DEL ESTUDIANTE

plt.scatter(x_2[:,0], x_2[:,1], c ='r') #pintamos dicha muestra
plt.show()


###############################################################################
###############################################################################
###############################################################################


# EJERCICIO 1.2: Dibujar una gráfica con la nube de puntos de salida correspondiente

# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):
	if x >= 0:
		return 1
	return -1

def f(x, y, a, b):
	return signo(y - a*x - b)

#CODIGO DEL ESTUDIANTE


input("\n--- Pulsar tecla para continuar ---\n")

# 1.2.b. Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto con la recta usada para ello
# Array con 10% de indices aleatorios para introducir ruido

#CODIGO DEL ESTUDIANTE

x_3 = simula_unif(100, 2, [-50,50])
etiquetas=[]

a,b = simula_recta([-50,50])
for i in range(0,len(x_3)): # asignamos a cada elemnto su etiqueta mediante la funcion g
    etiquetas.append(f(x_3[i,0], x_3[i,1], a,b))
    
etiquetas = np.asarray(etiquetas)

t = np.linspace(min(x_3[:,0]),max(x_3[:,0]), 100)
plt.scatter(x_3[:,0], x_3[:,1], c =etiquetas) #pintamos dicha muestra
plt.plot( t, a*t+b, c = 'red')
plt.show()


input("\n--- Pulsar tecla para continuar ---\n")

positivas = np.where(etiquetas == 1)
negativas = np.where(etiquetas == -1)
positivas = np.asarray(positivas).T
negativas = np.asarray(negativas).T


ind_pos = np.random.choice(len(positivas), int(0.1*len(positivas)), replace = True)

cambiar_signo = positivas[ind_pos,:]
ind_neg = np.random.choice(len(negativas), int(0.1*len(negativas)), replace = True)

cambiar_signo = np.concatenate((cambiar_signo, negativas[ind_neg,:]), axis=0)

for i in range(0, len(cambiar_signo)):
    etiquetas[cambiar_signo[i]]=-etiquetas[cambiar_signo[i]] 
    
t = np.linspace(min(x_3[:,0]),max(x_3[:,0]), 100)
plt.scatter(x_3[:,0], x_3[:,1], c =etiquetas) #pintamos dicha muestra
plt.plot( t, a*t+b, c = 'red')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")
###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 1.3: Supongamos ahora que las siguientes funciones definen la frontera de clasificación de los puntos de la muestra en lugar de una recta

def g1(x,y):
    return signo((x-10)**2+(y-20)**2-400)

def g2(x,y):
    return signo(0.5*(x+10)**2+(y-20)**2-400)

def g3(x,y):
    return signo(0.5*(x-10)**2-(y+20)**2-400)

def g4(x,y):
    return signo(y-20*x**2-5*x+3)


x_4 = simula_unif(100, 2, [-50,50])
etiquetas=[]


for i in range(0,len(x_4)): # asignamos a cada elemnto su etiqueta mediante la funcion g
    etiquetas.append(g1(x_4[i,0], x_4[i,1]))
    
etiquetas = np.asarray(etiquetas)

t = np.linspace(min(x_4[:,0]),max(x_4[:,0]), 100)
plt.scatter(x_4[:,0], x_4[:,1], c =etiquetas) #pintamos dicha muestra
plt.plot( t, np.sqrt(-t**2+20*t+300)+20, c = 'red')
plt.plot(t, 20-np.sqrt(-t**2+20*t+300), c = 'red')
plt.show()



input("\n--- Pulsar tecla para continuar ---\n")

etiquetas=[]


for i in range(0,len(x_4)): # asignamos a cada elemnto su etiqueta mediante la funcion g
    etiquetas.append(g2(x_4[i,0], x_4[i,1]))
    
etiquetas = np.asarray(etiquetas)

t = np.linspace(min(x_4[:,0]),max(x_4[:,0]), 100)
plt.scatter(x_4[:,0], x_4[:,1], c =etiquetas) #pintamos dicha muestra
plt.plot( t, 1/2*(40-np.sqrt(2)*np.sqrt(-t**2-20*t+700)), c = 'red')
plt.plot(t, 1/2*(40+np.sqrt(2)*np.sqrt(-t**2-20*t+700)), c = 'red')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

etiquetas=[]


for i in range(0,len(x_4)): # asignamos a cada elemnto su etiqueta mediante la funcion g
    etiquetas.append(g3(x_4[i,0], x_4[i,1]))
    
etiquetas = np.asarray(etiquetas)

t = np.linspace(min(x_4[:,0]),max(x_4[:,0]), 100)
plt.scatter(x_4[:,0], x_4[:,1], c =etiquetas) #pintamos dicha muestra
plt.plot( t, 1/2*(-40-np.sqrt(2)*np.sqrt(t**2-20*t-700)), c = 'red')
plt.plot(t,  1/2*(-40+np.sqrt(2)*np.sqrt(t**2-20*t-700)), c = 'red')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")


etiquetas=[]


for i in range(0,len(x_4)): # asignamos a cada elemnto su etiqueta mediante la funcion g
    etiquetas.append(g4(x_4[i,0], x_4[i,1]))
    
etiquetas = np.asarray(etiquetas)


t = np.linspace(min(x_4[:,0]),max(x_4[:,0]), 100)
posiciones_dominio = np.where(20*t**2+5*t-3 < max(x_4[:,1]))
t = t[posiciones_dominio]
plt.scatter(x_4[:,0], x_4[:,1], c =etiquetas) #pintamos dicha muestra
plt.plot( t,20*t**2+5*t-3, c = 'red')

plt.show()