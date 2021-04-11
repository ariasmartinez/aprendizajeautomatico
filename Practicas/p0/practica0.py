#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 16:25:18 2021

@author: Celia Arias

Práctica 0

"""


import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
import math 

#Ejercicio 1

iris = datasets.load_iris()
x = iris.data[:,::2]
y = iris.target

#cargamos los datos de la base de datos e introducimos en x las características 1 y 3 y en y las clases


colors = ['black', 'orange', 'green']
y0 = np.where(y == 0)
y1 = np.where(y == 1)
y2 = np.where(y == 2)
#Hacemos 3 arrays para separar los indices de las clases 0,1 y 2
x_2 = np.array([x[y0],x[y1],x[y2]])

for i in range (0,3):
        plt.scatter(x_2[i][:, 0], x_2[i][:, 1],  c = colors[i], label = iris.target_names[i])
    
#Dibujamos la gráfica coloreando cada clase con un color diferente y poniéndole su etiqueta


         
plt.xlabel("Longitud de Sépalo")
plt.ylabel("Longitud de Pétalo")
plt.legend();
plt.title("Ejercicio1")


plt.show()

#Ejercicio 2

#documentación: https://realpython.com/train-test-split-python-data/
x_train, x_test = train_test_split(x, test_size=0.25, stratify=y)

print("Training", x_train)
print("Test", x_test)


#Ejercicio 3

#documentación: https://iaarhub.github.io/capacitacion/2017/04/07/mini-tutorial-de-numpy/


a = np.linspace(0,4*math.pi,100)
print('Valores equiespaciados entre 0 y 4pi', a)
print('Seno de los valores del array')
b = np.sin(a)
print(b)
print('Coseno de los valores del array')
c = np.cos(a)
print(c)
print('Tanh de los valores del array')
d = np.tanh(np.cos(a)+np.sin(a))
print(d)

plt.plot(b, 'g--', label = 'sin')
plt.plot(c, 'b--', label = 'cos')
plt.plot(d, 'r--', label = 'tanh(sinx + cosx')
plt.legend();
plt.title("Ejercicio3")
plt.show()






