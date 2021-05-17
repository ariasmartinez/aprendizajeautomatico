#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 15:46:18 2021

@author: Celia Arias Martínez
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.manifold import TSNE

"""
readData : método para leer datos de un fichero
        nombre_fichero: nombre del fichero
        
        x: matriz de características
        y : vector con las etiquetas

"""
def readData (nombre_fichero):
    
    #función de la biblioteca de pandas para leer datos. El parámetro sep es el 
    #delimitador que utilizamos y header son las filas que se utilizan para los nombres
    # de las variables, en este caso ninguna
    data = pd.read_csv(nombre_fichero,
                       sep = ' ',
                       header = None)
    values = data.values

    # Nos quedamos con todas las columnas salvo la última (la de las etiquetas)
    x = values [:: -1]
    y = values [:, -1] # guardamos las etiquetas

    return x,y


x, y = readData("./data/clasificacion/Sensorless_drive_diagnosis.txt")


"""
parametros tsne:
    n_components 
    perplexity (30) : valor entre 5 y 50 (mayor cuanto mayor sea el dataset)
    early_exaggeration (12 por defecto): Este parámetro controla la distancia entre bloques semejantes en el espacio final. La elección de este valor no es crítico.
    learning_rate (200 por defecto): Habitualmente en el rango (10-1000). Si es muy elevado, los datos transformados estarán formados por una bola de puntos equidistantes unos de otros. Si es muy bajo, los puntos se mostrarán comprimidos en una densa nube con algunos outliers.
    n_iter (1000 por defecto): Número máximo de iteraciones para la optimización. Debería ser, por lo menos, 250.
    metric: métrica para la medición de las distancias.
    method: algoritmo a usar para el cálculo del gradiente.
    

"""

tsne = TSNE(n_components = 2, perplexity = 30)
x_tsne = tsne.fit_transform(x)
#x_tsne = TSNE(n_components=2).fit_transform(x)
plt.scatter(x_tsne[:, 0], x_tsne[:, 1],  c = y)
plt.show()

"""
tsne = TSNE(early_exaggeration = 30, init = "pca",
                        random_state = SEED, n_jobs = -1)
            X_new = tsne.fit_transform(X_train)
            for c in np.unique(y_train):
                idx = np.where(y_train == c)
                plt.scatter(X_new[idx, 0], X_new[idx, 1], label = c)
            plt.title("Represntación 2D mediante t-SNE")
            plt.xlabel("1ª dimensión")
            plt.ylabel("2ª dimensión")
            plt.legend(ncol = 2, fontsize = "x-small")
            plt.show()
            plt.close()

"""
tsne = TSNE(n_components = 2, perplexity = 45)
x_tsne = tsne.fit_transform(x)
#x_tsne = TSNE(n_components=2).fit_transform(x)
plt.scatter(x_tsne[:, 0], x_tsne[:, 1],  c = y)
plt.show()

tsne = TSNE(n_components = 2, learning_rate = 400)
x_tsne = tsne.fit_transform(x)
#x_tsne = TSNE(n_components=2).fit_transform(x)
plt.scatter(x_tsne[:, 0], x_tsne[:, 1],  c = y)
plt.show()
input("\n--- Pulsar tecla para continuar ---\n")

