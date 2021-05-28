#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 11:53:55 2021

@author: Celia Arias Martínez

REGRESIÓN
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 15:46:18 2021

@author: Celia Arias Martínez
"""

import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Perceptron
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
SEED = 42

"""
readData : método para leer datos de un fichero
        nombre_fichero: nombre del fichero
        
        x: matriz de características
        y : vector con las etiquetas

"""
def readData (nombre_fichero, cabecera = None):
    
    #función de la biblioteca de pandas para leer datos. El parámetro sep es el 
    #delimitador que utilizamos y header son las filas que se utilizan para los nombres
    # de las variables, en este caso ninguna
    data = pd.read_csv(nombre_fichero,
                       sep = ',',
                       header = cabecera)
    values = data.values
    
    # Nos quedamos con todas las columnas salvo la última (la de las etiquetas)
    x = values [:,:-1]
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
x, y = readData("./data/regresion/train.csv",0)   
#w,z = readData("./data/regresion/unique_m.csv",0)
print(x)

#x = np.concatenate((u,w), axis = 0)
#y = np.concatenate((v,z), axis = 0)


input("\n--- Pulsar tecla para continuar ---\n")

print("Separamos en test y training y vemos que los conjuntos están balanceados")


#dividimos en test y training
x_train, x_test, y_train_unidime, y_test_unidime = train_test_split(x, y, test_size = 0.2, random_state = SEED)


#vemos que están balanceados
y_train = y_train_unidime.reshape((-1,1))
y_aux = np.concatenate((y_train, np.zeros((len(y_train_unidime),1))), axis = 1)

plt.plot(y_aux[:,0],y_aux[:,1],'o')
plt.show()
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
    dibuja_tsne(x_train, y_train_unidime,2 )


input("\n--- Pulsar tecla para continuar ---\n")
#primero convertimos nuestra matriz en un data frame de pandas

y_train = y_train_unidime.reshape(-1,1)
y_test = y_test_unidime.reshape(-1,1)
df_train = pd.DataFrame(np.concatenate((x_train, y_train), axis = 1))
df_test =pd.DataFrame(np.concatenate((x_test, y_test), axis = 1))
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

def get_top_abs_correlations(matriz_corr, df):
    au_corr = matriz_corr.unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    strong_pairs = au_corr[abs(au_corr) > 0.9]
    return strong_pairs



#Pintamos la matriz de correlación pero solo con los pares que tengan un coeficiente de Pearson >= 0.9
correlation_mat = df_train.corr().abs()
correlation_mat = correlation_mat[correlation_mat > 0.95]
plt.figure(figsize=(12,8))
sns.heatmap(correlation_mat, cmap="Greens")
plt.show()

#https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas
#nos quedamos con los pares que tengan coeficiente mayor que 0.9, quitando la redundancia
print("Parejas con coeficiente de correlación de Pearson mayor que 0.9")
correlaciones = get_top_abs_correlations(correlation_mat, df_train)
print(correlaciones)

#Los resultados son: 0,2,5,6,7,11,12,15,17,22,26,25,27,33,37,47,52,57,67,69,71,72,77


#Eliminamos los atributos (23, nos quedan 58)

input("\n--- Pulsar tecla para continuar ---\n")



#eliminar datos sin variabilidad

df_train.boxplot(figsize=(12,6), rot=90, column=list(df_train.columns[:int(len(df_train.columns))]))
plt.show()

#Vemos que las variables que tienen variabilidad casi nula son la 20,69,70
#Por tanto elimino los atributos 20,70 (los demás estaban eliminados antes)

input("\n--- Pulsar tecla para continuar ---\n")
#reducir la dimensionalidad
df_train.drop([0,2,5,6,7,11,12,15,17,20,22,26,25,27,33,37,47,52,57,67,69,70,71,72,77],axis=1)
df_test.drop([0,2,5,6,7,11,12,15,17,20,22,26,25,27,33,37,47,52,57,67,69,70,71,72,77],axis=1)
x_train_reduced = np.delete(x_train, [0,2,5,6,7,11,12,15,17,20,22,26,25,27,33,37,47,52,57,67,69,70,71,72,77],axis=1)
x_test_reduced= np.delete(x_test,[0,2,5,6,7,11,12,15,17,20,22,26,25,27,33,37,47,52,57,67,69,70,71,72,77],axis=1)

#Me quedan ahora 29 variables

#aplicamos PCA

#pca = PCA(0.99)
#x_train_reduced = pca.fit_transform(x_train_reduced)
#x_test_reduced = pca.transform( x_test_reduced)
#x_train_reduced = x_train
#y_test_reduced = x_test

#varianza_explicada = np.asarray(pca.explained_variance_ratio_)
#print(varianza_explicada.sum)


input("\n--- Pulsar tecla para continuar ---\n")
#Volvemos a visualizar después de haber reducido el número de variables

if input("¿Quieres ver una representación de los datos usando t-SNE? (s/n): ") == "s":
   dibuja_tsne(x_train_reduced, y_train_unidime,2 )
    


input("\n--- Pulsar tecla para continuar ---\n")

#cambiar learning rate, tamaño mini bach, criterio de parada...
"""
cs: fuerza de regularizacion
cv : cross validation ??
penalty: regularizacion  PONER
score: metrica de evaluacion
solver: algoritmo a usar en el problema de optimizacion
tol: tolerancia para el criterio de parada
max_iter: maximo de iteraciones
multi_class : 
random_state: para mezclar los datos

"""




modelos = [SGDRegressor(loss=algoritmo, penalty=pen, alpha=a, learning_rate = lr, eta0 = 0.01, max_iter=5000) for a in [0.0001,0.001] for algoritmo in ['squared_loss', 'epsilon_insensitive'] for pen in ['l1', 'l2'] for lr in ['optimal', 'adaptive'] ]


start_time = time()



best_score = 0
for model in modelos:
    print(model)
    score = np.mean(cross_val_score(model, x_train_reduced, y_train_unidime, cv = 5, scoring="r2",n_jobs=-1))
    print(score)
    #plot_confusion_matrix(model, x_train_reduced, y_train_unidime)
    if best_score < score:
           best_score = score
           best_model = model
    

print("Hacemos el entrenamiento")
print(best_model)
best_model.fit(x_train_reduced, y_train_unidime)

print("Hacemos prediccion")
y_pred_logistic = best_model.predict(x_test_reduced)
y_pred_logistic_train = best_model.predict(x_train_reduced)
print("Calculamos coeficientes de determinación")

coef_regres_test = best_model.score(x_test_reduced, y_test_unidime)
coef_regres_train = best_model.score(x_train_reduced, y_train_unidime)
print("\tCoeficiente de determinación en test: ", coef_regres_test)
print("\tCoeficiente de determinación en entrenamiento: ", coef_regres_train)

#y_aleatorio = np.random.randint(0,11,len(y_test))
#numero_aciertos_aleatorio = accuracy_score(y_test,y_aleatorio)
#print("\tPorcentaje de aciertos de forma aleatoria: ", numero_aciertos_aleatorio)
#print(100*best_model.score(x_train_trans, y_train_unidime))
#print(100* best_model.score(x_test_trans, y_test_unidime))


input("\n--- Pulsar tecla para continuar ---\n")
Etest=mean_squared_error(y_test_unidime, y_pred_logistic)
print("Error cuadratico medio en test: ",Etest)
Etrain=mean_squared_error(y_train_unidime, y_pred_logistic_train)
print("Error cuadratico medio en test: ",Etest)
#print(100* best_model.score(x_test_reduced, y_test_unidime))
input("\n--- Pulsar tecla para continuar ---\n")
print("Matriz de confusión")

#plot_confusion_matrix(best_model, x_train_reduced, y_train_unidime)
#plt.show()
#plot_confusion_matrix(best_model, x_test_reduced, y_test_unidime)

#plt.show()

input("\n--- Pulsar tecla para continuar ---\n")
#21/41/43/57/05/18/...

best_model.fit(x_train, y_train_unidime)
y_pred_logistic = best_model.predict(x_test)
coef_regres_test = best_model.score(x_test, y_test_unidime)
coef_regres_train = best_model.score(x_train, y_train_unidime)
print("\tCoeficiente de determinación en test: ", coef_regres_test)
print("\tCoeficiente de determinación en train: ", coef_regres_train)
