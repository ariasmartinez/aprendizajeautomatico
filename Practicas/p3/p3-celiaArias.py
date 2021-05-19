#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 15:46:18 2021

@author: Celia Arias Martínez
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns

from sklearn.manifold import TSNE

SEED = 42

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



def dibuja_tsne(x,y, n_components, perpl = 30, early_exag = 12, lr = 200):
    tsne = TSNE(n_components = 2, perplexity = perpl,early_exaggeration= early_exag, learning_rate = lr )
    x_tsne = tsne.fit_transform(x)
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1],  c = y)
    plt.show()

def categorias_balanceadas(num_categorias, y):
    cantidades = np.zeros(num_categorias)
    for i in y: #posición i de cantidad es la categoría i+1
        cantidades[int(i)-1]+=1
    
    return cantidades/float(len(y))


print("Leemos los datos")
x, y = readData("./data/clasificacion/Sensorless_drive_diagnosis.txt")   

input("\n--- Pulsar tecla para continuar ---\n")

print("Separamos en test y training y vemos que los conjuntos están balanceados")
num_categorias = 11

#dividimos en test y training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = SEED)
#vemos que están balanceados
cantidades_proporcion = categorias_balanceadas(num_categorias, y_train)
print("Proporción de elementos en cada categoría en el conjunto de entrenamiento :" , cantidades_proporcion)
cantidades_proporcion = categorias_balanceadas(num_categorias, y_test)
print("Proporción de elementos en cada categoría en el conjunto de test:" , cantidades_proporcion)

input("\n--- Pulsar tecla para continuar ---\n")

print("Normalizamos los datos")
#Normalizamos los datos para que tengan media 0 y varianza 1
scaler = StandardScaler()
x_train = scaler.fit_transform( x_train )
x_test = scaler.transform( x_test)


input("\n--- Pulsar tecla para continuar ---\n")
#visualizamos los datos en 2-d
#TSNE con parámetros por defecto
if input("¿Quieres ver una representación de los datos usando t-SNE? (s/n): ") == "s":
    dibuja_tsne(x_train, y_train,2 )


input("\n--- Pulsar tecla para continuar ---\n")
#primero convertimos nuestra matriz en un data frame de pandas

y_train = y_train.reshape(-1,1)
df_train = pd.DataFrame(np.concatenate((x_train, y_train), axis = 1))

print("Vemos si el dataset tiene valores perdidos (True significa que no hay valores perdidos)")
print(np.all(df_train.notnull()))


input("\n--- Pulsar tecla para continuar ---\n")
#Estudiamos la matriz de correlación, para ver si podemos eliminar atributos

def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]



#calculamos la matriz de correlación  (calcula el coeficiente de correlación de Pearson)
#correlation_mat = df_train.corr()
#sns.heatmap(correlation_mat, annot = True) #la visualizamos con colores
#plt.show()

#la matriz es demasiado grande para poder ver a simple vista qué valores están correlacionados
#Para trabajar con ella convertimos la matriz dada en una serie unidimensional de valores.
#corr_pairs = correlation_mat.unstack()

#ordenamos los valores
#sorted_pairs = corr_pairs.sort_values(kind="quicksort")

#sorted_pairs = sorted_pairs[sorted_pairs[0] != sorted_pairs[1]]
#strong_pairs = sorted_pairs[abs(sorted_pairs) > 0.5]


#RESULTADOS
#Vemos que las columnas 48 y 49 son inversamente proporcionales, así que eliminamos una 
#de las dos (en este caso la 49)
"""
48  49    1.0
21  22    1.0
18  19    1.0

"""
#https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas
print("Top Absolute Correlations")
print(get_top_abs_correlations(df_train, 1000))


input("\n--- Pulsar tecla para continuar ---\n")

#cross validation

input("\n--- Pulsar tecla para continuar ---\n")
#eliminar datos sin variabilidad

input("\n--- Pulsar tecla para continuar ---\n")
#reducir la dimensionalidad

input("\n--- Pulsar tecla para continuar ---\n")
#datos extremos

input("\n--- Pulsar tecla para continuar ---\n")
#regularizacion