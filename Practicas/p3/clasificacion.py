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
from sklearn.model_selection import cross_val_score
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import plot_confusion_matrix
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
    x = values [:,:-1]
    y = values [:, -1] # guardamos las etiquetas

    return x,y


"""
dibuja_tsne : función para dibujar en 2d utilizando t-SNE
    x: matriz de características
    y: vector de etiquetas
    perpl: perplexity, 30 por defecto
    early_exag : early_exaggeration, por defecto 12
    lr: learning_rate, por defecto 200
"""
    
def dibuja_tsne(x,y, perpl = 30, early_exag = 12, lr = 200):
    tsne = TSNE(n_components = 2, perplexity = perpl,early_exaggeration= early_exag, learning_rate = lr )
    x_tsne = tsne.fit_transform(x)
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1],  c = y)
    plt.show()


"""
categorias_balanceadas: función para ver si un vector tiene un número equitativo de etiquetas de cada clase
    num_categorías: número de clases
    y: vector de etiquetas
    
    return: vector con la proporción de cada clase
"""
def categorias_balanceadas(num_categorias, y):
    cantidades = np.zeros(num_categorias)
    for i in y: #posición i de cantidad es la categoría i+1
        cantidades[int(i)-1]+=1
    
    return cantidades/float(len(y))


print("Leemos los datos")
x, y = readData("./datos/Sensorless_drive_diagnosis.txt")   
input("\n--- Pulsar tecla para continuar ---\n")

print("Separamos en test y training y vemos que los conjuntos están balanceados")
num_categorias = 11

#dividimos en test y training
x_train, x_test, y_train_unidime, y_test_unidime = train_test_split(x, y, test_size = 0.2, random_state = SEED)

#vemos que están balanceados
cantidades_proporcion = categorias_balanceadas(num_categorias, y_train_unidime)
print("Proporción de elementos en cada categoría en el conjunto de entrenamiento :" , cantidades_proporcion)

input("\n--- Pulsar tecla para continuar ---\n")

print("Normalizamos los datos")
#Normalizamos los datos para que tengan media 0 y varianza 1
scaler = StandardScaler()
x_train = scaler.fit_transform( x_train )
x_test = scaler.transform( x_test)



#visualizamos los datos en 2-d
#TSNE con parámetros por defecto
#dibuja_tsne(x_train, y_train_unidime )


input("\n--- Pulsar tecla para continuar ---\n")
#Vemos que no faltan datos 

#primero convertimos nuestra matriz en un data frame de pandas
y_train = y_train_unidime.reshape(-1,1)
y_test = y_test_unidime.reshape(-1,1)
df_train = pd.DataFrame(np.concatenate((x_train, y_train), axis = 1))
df_test =pd.DataFrame(np.concatenate((x_test, y_test), axis = 1))
print("Vemos si el dataset tiene valores perdidos (True significa que no hay valores perdidos)")
print(np.all(df_train.notnull()))


input("\n--- Pulsar tecla para continuar ---\n")
#Estudiamos la matriz de correlación, para ver si podemos eliminar atributos

#fuente auxiliar: https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(matriz_corr, df):
    au_corr = matriz_corr.unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    strong_pairs = au_corr[abs(au_corr) > 0.9]
    return strong_pairs



#Pintamos la matriz de correlación pero solo con los pares que tengan un coeficiente de Pearson > 0.9
print("Gráfica con la matriz de correlación:")
correlation_mat = df_train.corr().abs()
correlation_mat = correlation_mat[correlation_mat > 0.9]
plt.figure(figsize=(12,8))
sns.heatmap(correlation_mat, cmap="Greens")
plt.show()


#nos quedamos con los pares que tengan coeficiente mayor que 0.9, quitando la redundancia
print("Parejas con coeficiente de correlación de Pearson mayor que 0.9")
print(get_top_abs_correlations(correlation_mat, df_train))



#Eliminamos los atributos 7,8,10,11,13,16,19,20,21,22,23,31,32,34,35,43,44,46,47


input("\n--- Pulsar tecla para continuar ---\n")
#eliminar datos sin variabilidad
print("Estudiamos la variabilidad")
df_train.boxplot(figsize=(12,6), rot=90, column=list(df_train.columns[:int(len(df_train.columns))]))
plt.show()

#Vemos que las variables que tienen variabilidad casi nula son la 18,19,20,21,22 y 23
#Por tanto elimino el atributo 18 (los demás estaban eliminados antes)

#reducir la dimensionalidad
df_train.drop([7,8,10,11,13,16,18,19,20,21,22,23,31,32,34,35,43,44,46,47],axis=1)
df_test.drop([7,8,10,11,13,16,18,19,20,21,22,23,31,32,34,35,43,44,46,47],axis=1)
x_train_reduced = np.delete(x_train, [7,8,10,11,13,16,18,19,20,21,22,23,31,32,34,35,43,44,46,47],axis=1)
x_test_reduced= np.delete(x_test, [7,8,10,11,13,16,18,19,20,21,22,23,31,32,34,35,43,44,46,47],axis=1)

#Me quedan ahora 29 variables


#Volvemos a visualizar después de haber reducido el número de variables
#dibuja_tsne(x_train_reduced, y_train_unidime )
    


input("\n--- Pulsar tecla para continuar ---\n")
print("Entrenamos los modelos (Puede tardar unos minutos -alrededor de 2-)")
#modelos contiene la selección de los modelos que hemos hecho, de donde elegiremos el mejor para entrenar nuestros datos

modelos = [SGDClassifier(loss=algoritmo, penalty=pen, alpha=a, learning_rate = lr, eta0 = 0.01, max_iter=10000, n_jobs = -1) for a in [0.0001,0.001] for algoritmo in ['hinge', 'log'] for pen in ['l1', 'l2'] for lr in ['optimal', 'adaptive'] ]


#Recorremos todos los modelos, calculamos la media de accuracy de cada uno con cross-validation y nos quedamos con el mejor
best_score = 0
for model in modelos:
    score = np.mean(cross_val_score(model, x_train_reduced, y_train_unidime, cv = 5, scoring="accuracy",n_jobs=-1))
    if best_score < score:
           best_score = score
           best_model = model
    

print("Entrenamos nuestros datos con el modelo seleccionado")
print(best_model.get_params())
best_model.fit(x_train_reduced, y_train_unidime)

print("Hacemos prediccion al conjunto de test y de train")
y_pred_logistic = best_model.predict(x_test_reduced)
y_pred_logistic_train = best_model.predict(x_train_reduced)

print("Calculamos accuracy")
numero_aciertos_test = accuracy_score(y_test, y_pred_logistic)
numero_aciertos_train = accuracy_score(y_train, y_pred_logistic_train)
print("\tPorcentaje de aciertos en test: ", numero_aciertos_test)
print("\tPorcentaje de aciertos en entrenamiento: ", numero_aciertos_train)

#Generamos las etiquetas de forma aleatoria y vemos el porcentaje de aciertos que obtenemos
y_aleatorio = np.random.randint(0,11,len(y_test))
numero_aciertos_aleatorio = accuracy_score(y_test,y_aleatorio)
print("\tPorcentaje de aciertos de forma aleatoria: ", numero_aciertos_aleatorio)


input("\n--- Pulsar tecla para continuar ---\n")
print("Matriz de confusión en entrenamiento")

plot_confusion_matrix(best_model, x_train_reduced, y_train_unidime)
plt.show()

print("Matriz de confusión en test")
plot_confusion_matrix(best_model, x_test_reduced, y_test_unidime)
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")
print("Entrenamos con toda la muestra, sin hacer reducción")

best_model.fit(x_train, y_train_unidime)
y_pred_logistic = best_model.predict(x_test)
y_pred_logistic_train= best_model.predict(x_train)
numero_aciertos_test = accuracy_score(y_test, y_pred_logistic)
numero_aciertos_train = accuracy_score(y_train, y_pred_logistic_train)


print("\tPorcentaje de aciertos en test: ", numero_aciertos_test)
print("\tPorcentaje de aciertos en train: ", numero_aciertos_train)