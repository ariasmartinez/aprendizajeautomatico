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

def calculaPorcentaje(x, y, g):
    
    mal_etiquetadas = 0
    for i in range(0,len(x[:,0])):
        etiqueta_real = y[i]
        etiqueta_obtenida = g(x[i,0], x[i,1])
        if (etiqueta_real != etiqueta_obtenida):
            mal_etiquetadas+=1

    porcentaje_mal = mal_etiquetadas / float(len(x))
    
    return porcentaje_mal


def aux(x,y):  # w0+ w1*x1+ w1*x2 = 0
    return signo(w[0]+w[1]*x+w[2]*y)


"""
sgdRL: algoritmo de regresión logística con N = 1
    x : matriz de características
    y: etiquetas
    max_iter: número máximo de iteraciones que puede realizar
    eta: learning rate
    epsilon: cota de error que permitimos
    
    w.T: vector de pesos encontrado
    iteraciones: número de iteraciones realizado

"""
def sgdRL(x, y, max_iter, eta ,epsilon):
    w = np.zeros((x[0].size,1)).reshape((1,-1)) #vector inicial = 0
    y = y.reshape((-1, 1))  #convertimos y en un vector columna para poder realizar la multiplicación
    iteraciones= 0 
    indices = np.arange(len(y)) #vector con los números entre 0 y el tamaño de y
    for i in range (0, max_iter): 
        iteraciones+=1
        np.random.shuffle(indices) #aplicamos una permutación aleatoria a los índices
        w_old = np.copy(w) #guardamos el w anterior
        for j in indices: #recorremos todos los datos
            exponencial = y[j]*((w.dot(x[j,:]))) 
            grad = -(y[j]*(x[j,:]))/(1+math.e**(exponencial)) 
            w = w -eta*grad
        
        #si hay muy poca diferencia entre un w y el anterior paramos el algoritmo
        if (np.linalg.norm(w_old-w) < epsilon): break
       
    return w.T, iteraciones


#simulamos una muestra de tamaño 100
tamaño = 100
x_entre = simula_unif(tamaño, 2, [0,2]) 
indice = np.random.randint(0,tamaño, (2,1)) #tomamos dos indices aleatorios de dos puntos de la muestra
punto_a =[ x_entre[indice[0],0], x_entre[indice[0],1]] #tomamos un primer punto con el primer índice
punto_b =[ x_entre[indice[1],0], x_entre[indice[1],1]] #tomamos un segundo punto con el segundo índice

a = (punto_b[1]-punto_b[0])/(punto_a[1]-punto_b[0]) # Calculo de la pendiente.
b = punto_b[0] - a*punto_a[0]       # Calculo del termino independiente.

"función para etiquetar los puntos de la muestra"
def h(x,y):
    return signo(y-a*x-b)

etiquetas_entre = []
for i in range(0,len(x_entre)): # asignamos a cada elemnto su etiqueta mediante la funcion h
    etiquetas_entre.append(h(x_entre[i,0], x_entre[i,1]))

etiquetas_entre = np.asarray(etiquetas_entre) #convertimos etiquetas en un arreglo

t = np.linspace(min(x_entre[:,0]),max(x_entre[:,0]), 100) #generamos 100 puntos entre mímino punto de la muestra y el máximo
plt.ylim(0, 2) #para que la recta no se salga del dominio
plt.scatter(x_entre[:,0], x_entre[:,1], c =etiquetas_entre) #pintamos dicha muestra, diferenciando los colores por las etiquetas
plt.plot( t, a*t+b, c = 'red') #pintamos la recta de rojo
plt.show()




input("\n--- Pulsar tecla para continuar ---\n")
    
#Ahora aprendemos la recta frontera con el algoritmo de regresión logística   

#preparamos los datos para poder aplicar el algoritmo añadiendo una columna de unos 
vector_unos = np.ones((len(x_entre),1))
datos_entre = np.copy(x_entre)
datos_entre = np.concatenate((vector_unos, datos_entre), axis = 1)
max_iteraciones = 1000
eta = 0.01
epsilon = 0.01
w, it = sgdRL(datos_entre, etiquetas_entre, max_iteraciones, eta, epsilon)
 
   
t = np.linspace(min(x_entre[:,0]),max(x_entre[:,0]), 100) #generamos 100 puntos entre mímino punto de la muestra y el máximo
plt.scatter(x_entre[:,0], x_entre[:,1], c =etiquetas_entre) #pintamos dicha muestra, diferenciando los colores por las etiquetas
plt.plot( t, a*t+b, c = 'green', label = 'frontera') #pintamos la recta frontera de verde
plt.plot( t, (-w[0]-w[1]*t)/w[2], c = 'red', label = 'sgdRL') #pintamos la recta de rojo
plt.ylim(0, 2)
plt.legend()
plt.show()

print('w: ', w)
print('Número de iteraciones: ', it)
print('Porcentaje mal etiquetados: ',calculaPorcentaje(x_entre,etiquetas_entre,h) )


input("\n--- Pulsar tecla para continuar ---\n")

tamaño_test = 1000
x_test = simula_unif(tamaño_test, 2, [0,2]) 
#añadimos una columna a la matriz para después poder calcular Eout
vector_unos = np.ones((len(x_test),1))
datos_test = np.copy(x_test)
datos_test = np.concatenate((vector_unos, datos_test), axis = 1)


etiquetas_test = []
for i in range(0,len(x_test)): # asignamos a cada elemnto su etiqueta mediante la funcion h
    etiquetas_test.append(h(x_test[i,0], x_test[i,1]))
    
etiquetas_test = np.asarray(etiquetas_test) #convertimos etiquetas en un arreglo

t = np.linspace(min(x_test[:,0]),max(x_test[:,0]), 100) #generamos 100 puntos entre mímino punto de la muestra y el máximo
plt.ylim(0, 2) #para que la recta no se salga del dominio
plt.scatter(x_test[:,0], x_test[:,1], c =etiquetas_test) #pintamos dicha muestra, diferenciando los colores por las etiquetas
plt.plot( t, (-w[0]-w[1]*t)/w[2], c = 'red', label = 'sgdRL') #pintamos la recta de rojo
plt.plot( t, a*t+b, c = 'green', label = 'frontera') #pintamos la recta frontera de verdeÇ
plt.legend()
plt.show()

print('Porcentaje mal etiquetados: ',calculaPorcentaje(x_test,etiquetas_test,aux) )

#calculamos Eout
print('Eout :', np.mean(np.log(1 + np.exp(-datos_test.dot(w) * etiquetas_test)))   )