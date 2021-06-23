#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 15:50:44 2021

@author: celia
"""
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt



iris = datasets.load_iris()
x = iris.data[:,::2]
Y = iris.target

#Iremos representando los datos en la gráfica por etiquetas
idx=np.where (Y==1); #tomo los índices con etiqueta 1 en este caso
Y0=Y[idx];  #me quedo con los valores en esos índices (todos los 1 de Y)
X0=X_1[idx]; #me quedo con los datos correspondientes a los elementos con etiqueta 1

#Repito el proceso con el resto de etiquetas

idx=np.where (Y==2);
Y1=Y[idx];
X1=X_1[idx];

idx=np.where (Y==3);
Y2=Y[idx];
X2=X_1[idx];

#Hago los gráficos de puntos con cada pareja de datos especificando los colores del ejercicio
plt.scatter(X0[:,0],X0[:,1], c='orange', label='Iris Setosa')
plt.scatter(X1[:,0],X1[:,1], c='black', label='Iris Versicolor')
plt.scatter(X2[:,0],X2[:,1], c='green', label='Iris Virginica')


plt.xlabel("Longitud de Sépalo");
plt.ylabel("Longitud de Pétalo");
plt.legend();

#Los muestro todos juntos
plt.show();