#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 16:34:54 2021

@author: celia
"""

import numpy as np
import matplotlib.pyplot as plt
import math


# Fijamos la semilla
np.random.seed(1)
def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))


def signo(x):
	if x >= 0:
		return 1
	return -1


""" N = 1 """
def sgdRL(x, y, max_iter, epsilon ,eta):
    w = np.zeros((x[0].size,1)).T
    y = y.reshape((1, -1))  #convertimos y en un vector columna para poder realizar la multiplicación
    #for i in range (0,10):
        print(w)
        w_old = np.copy(w)
        exponencial = y.dot((w.dot(x.T)).T)
        print(exponencial)
        if (exponencial > 500): exponencial = 500
        grad = -(y.dot(x))/(1+math.e**(exponencial))
        w = w_old -eta*grad
       # if ((w-w_old).max(axis = 1) < epsilon): break

    for j in range (0, max_iter):
        iteraciones+=1
        for i in range (0, len(etiquetas)):
            
    return w.T



x_entre = simula_unif(500, 2, [0,2]) 
indice = np.random.randint(0,50, (2,1))
punto_a =[ x_entre[indice[0],0], x_entre[indice[0],1]]
punto_b =[ x_entre[indice[1],0], x_entre[indice[1],1]]

a = (punto_b[1]-punto_b[0])/(punto_a[1]-punto_b[0]) # Calculo de la pendiente.
b = punto_b[0] - a*punto_a[0]       # Calculo del termino independiente.

def h(x,y):
    return signo(y-a*x-b)

etiquetas = []
for i in range(0,len(x_entre)): # asignamos a cada elemnto su etiqueta mediante la funcion f
    etiquetas.append(h(x_entre[i,0], x_entre[i,1]))

    
etiquetas = np.asarray(etiquetas) #convertimos etiquetas en un arreglo

t = np.linspace(min(x_entre[:,0]),max(x_entre[:,0]), 100) #generamos 100 puntos entre mímino punto de la muestra y el máximo
dominio = np.where(a*t+b > min(x_entre[:,0]))
t = t[dominio]
plt.scatter(x_entre[:,0], x_entre[:,1], c =etiquetas) #pintamos dicha muestra, diferenciando los colores por las etiquetas
plt.plot( t, a*t+b, c = 'red') #pintamos la recta de rojo
plt.show()


input("\n--- Pulsar tecla para continuar ---\n")
    
    
N = 100

vector_unos = np.ones((len(x_entre),1))
datos_entre = np.copy(x_entre)
datos_entre = np.concatenate((vector_unos, datos_entre), axis = 1)
w = sgdRL(datos_entre, etiquetas, 1000, 0.01, 0.005)
 
   
t = np.linspace(min(x_entre[:,0]),max(x_entre[:,0]), 100) #generamos 100 puntos entre mímino punto de la muestra y el máximo
plt.scatter(x_entre[:,0], x_entre[:,1], c =etiquetas) #pintamos dicha muestra, diferenciando los colores por las etiquetas
plt.plot( t, (-w[0]-w[1]*t)/w[2], c = 'red') #pintamos la recta de rojo
plt.show()

