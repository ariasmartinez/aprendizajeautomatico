#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 16:27:35 2021

@author: celia
"""

import matplotlib.pyplot as plt
from sklearn import datasets


# 1.1
iris = datasets.load_iris()   # leemos la base de datos iris

# 1.2 
print(iris.feature_names) # Con este comando podemos ver el nombre de las caracetristicas 
print (iris.target_names) # Nombre de las clases 


X = iris.data # obtenemos los datos
y = iris.target # obtenemos las clases


# 1.3 
X_filtrada = X[::2] # nos quedamos con primera y tercera, que son las pares por eso lo de paso 2



# 1.4

#_____- DIAGRAMA PARA SÉPALO ________
# Calculamos mínimo y máximo de los ejes para representar 
borde = .5
sepalo_longitud_indice = 0
sepalo_ancho_indice  = 1 

x_min, x_max = X[:, sepalo_longitud_indice].min() - borde, X[:, sepalo_longitud_indice].max() + borde
y_min, y_max = X[:, sepalo_ancho_indice ].min() - borde, X[:, sepalo_ancho_indice ].max() + borde

plt.figure("sepalo", figsize=(8, 6)) # identificador y tamaño 
plt.clf() # limpiamos buffer 

# naranja negro verde 
colores = ['#ff8000', '#000000', '#009f00']
y_colores = [ colores[i] for i in y] # asociamos a cada clase su color

plt.scatter(X[:, sepalo_longitud_indice], X[:, sepalo_ancho_indice], c=y_colores)

plt.xlabel('Sépalo longitud cm ')
plt.ylabel('Sépalo ancho cm')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)


plt.show()

#_____- DIAGRAMA PARA PÉTALO ________
# Calculamos mínimo y máximo de los ejes para representar 
borde = .5
petalo_longitud_indice = 2
petalo_ancho_indice  = 3 

x_min, x_max = X[:, petalo_longitud_indice].min() - borde, X[:, petalo_longitud_indice].max() + borde
y_min, y_max = X[:, petalo_ancho_indice ].min() - borde, X[:, petalo_ancho_indice ].max() + borde

plt.figure("petalo", figsize=(8, 6)) # identificador y tamaño 
plt.clf() # limpiamos buffer 

# naranja negro verde 
colores = ['#ff8000', '#000000', '#009f00']
y_colores = [ colores[i] for i in y] # asociamos a cada clase su color

plt.scatter(X[:, petalo_longitud_indice], X[:, petalo_ancho_indice], c=y_colores)

plt.xlabel('Pétalo longitud cm ')
plt.ylabel('Pétalo ancho cm')


plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.show()