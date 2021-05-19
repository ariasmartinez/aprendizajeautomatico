
---
title: Práctica 3
author: Celia Arias Martínez
header-includes: |
    \usepackage{tikz,pgfplots}
    \usepackage{fancyhdr}
    \pagestyle{fancy}
    \fancyhead[CO,CE]{Pŕactica 3}
    \fancyfoot[CO,CE]{Celia Arias Martínez}
    \fancyfoot[LE,RO]{\thepage}
   
    \usepackage[spanish,es-tabla]{babel}
    \usepackage[utf8]{inputenc}
    \usepackage{graphicx}
    \usepackage{subcaption}
---


En esta práctica vamos a realizar el ajuste y selección del mejor predictor lineal para un conjunto de datos dados. Vamos a tener dos problemas: uno de regresión y otro de clasificación, por lo que haremos dos secciones, y desarrollaremos dentro de cada sección los pasos que llevaremos a cabo, todos ellos encaminados a seleccionar el mejor modelo y la mejor estimación de error $E_{out}$

# Regresión

Para este problema utilizamos la base de datos *Superconductivty Data Data Set* encontrada en https://archive.ics.uci.edu/ml/datasets/Dataset+for+Sensorless+Drive+Diagnosis.


## Comprensión del problema a resolver. Identificación de los elementos $X, Y y f$.

El problema que queremos resolver es la asignación de una temperatura crítica a un superconductor, dadas unas características como el número de elementos, la masa atómica, el radio atómico, el punto de fusión, la entropía, etcétera. En concreto tenemos datos sobre 21263 superconductores, y de cada uno de ellos tenemos 81 características. 

Por tanto los elementos son:
$X$ : matriz con las características de los superconductores
$Y$: temperatura crítica del superconductor
$f$: función que asocia a cada superconductor su temperatura crítica.

Para comprender mejor el problema vamos a visualizar los datos. Para ello realizaremos dos pasos: reducir el conjunto de datos a un número de dimensiones razonable con la técnica PCA y visualizar el conjunto de datos con la técnica t-SNE.

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#Tenemos que escalar las características de nuestros datos antes de aplicar PCA.
#Normalizamos los datos para que tengan media 0 y varianza 1

x = StandardScaler().fit_transform(x)

#Aplicamos PCA
pca = PCA(n_components = 50)
principales = pca.fit_transform(x)
print(x[0])

## Selección de la clase de funciones a usar.


## Identificación de las hipótesis finales que vamos a usar.

## Partición en test y entrenamiento.

## Preprocesado de datos.

## Justificación de la métrica de error a usar.

## Justificación de los parámetros y del tipo de regularización usada.

## Selección de la mejor hipótesis del problema.

## Modelo final con todos los datos.




# Clasificación

Para este problema utilizamos la base de datos *Superconductivty Data Data Set* encontrada en https://archive.ics.uci.edu/ml/datasets/Dataset+for+Sensorless+Drive+Diagnosis.



## Comprensión del problema a resolver. Identificación de los elementos $X, Y y f$.

El problema que queremos resolver es dado un motor, asignarlo a una de las once clases que tenemos. Cada motor tiene unas características como DUDA. En concreto tenemos datos sobre 58509 motores, y de cada uno de ellos tenemos 49 características. 



Para comprender mejor el problema vamos a visualizar los datos. Para ello aplicaremos el algoritmo t-SNE. En este caso no hace falta aplicar antes PCA ya que tenemos un número de atributos inferior a 50 (tenemos 49). t-SNE intenta reproducir la distribución que existe en el espacio original en otro espacio de dimensión menor, en este caso de dimensión 2 para que podamos visualizarlo. Al contrario que PCA, que simplemente maximiza la varianza, t-SNE hace que puntos con características parecidas queden cerca en el modelo final, y los que menos se parecen queden alejados.

Adjunto el gráfico obtenido con t-SNE, se puede ejecutar en el código pero recomiendo no hacerlo, ya que tarda mucho. He intentado cambiar algunos parámetros para que vaya más rápido pero no ha funcionado ninguno, como explico más abajo.
\begin{figure}[!h]
\centering
\includegraphics[width=0.5\textwidth]{./graficas/defecto.png}
\caption{t-SNE parámetros por defecto}
\end{figure} 


t-SNE admite algunos parámetros tales como:

*  *perplexity* : valor entre 5 y 50 (mayor cuanto mayor sea el dataset). Por defecto vale 30.
* *early_exaggeration*: Este parámetro controla la distancia entre bloques semejantes en el espacio final. La elección de este valor no es crítico. Por defecto vale 12.
* *learning_rate*: Habitualmente en el rango (10-1000). Si es muy elevado, los datos transformados estarán formados por una bola de puntos equidistantes unos de otros. Si es muy bajo, los puntos se mostrarán comprimidos en una densa nube con algunos outliers. Poe defecto vale 200.
* *n_iter*: Número máximo de iteraciones para la optimización. Debería ser, por lo menos, 250. Por defecto vale 1000.
* *metric*: métrica para la medición de las distancias.
* *method*: algoritmo a usar para el cálculo del gradiente.

Vamos a cambiar algunos valores de los parámetros, para ver cómo influye en el resultado final y cuales de ellos se ajustan mejor a nuestro modelo.

\begin{figure}[h!]
\centering
\begin{subfigure}[b]{0.45\linewidth}
\includegraphics[width=\linewidth]{./graficas/perplexity=45.png}
\caption{perplexity = 45}
\end{subfigure}
\begin{subfigure}[b]{0.45\linewidth}
\includegraphics[width=\linewidth]{./graficas/lr=400.png}
\caption{learning rate = 400}
\end{subfigure}
\begin{subfigure}[b]{0.45\linewidth}
\includegraphics[width=\linewidth]{./graficas/early=30.png}
\caption{early-exaggeration=30}
\end{subfigure}
\end{figure} 

No he incluido estas gráficas en el código porque tardan mucho tiempo en ejecutar, pero las menciono aquí a modo de comentario.

Vemos que no tenemos diferencias muy significativas al variar los parámetros, al menos no tenemos diferencias que nos añadan más información de la que disponemos. En cuanto al tiempo podemos ver que el método tarda mucho tiempo en ejecutarse, pero tampoco he podido disminuir ese tiempo al cambiar los parámetros.


Por último he comprobado que las clases estén proporcionadas, es decir, que haya un número parecido de elementos de cada clase en la muestra. Los resultados han sido:

[0.0909 0.0909 0.0909 0.0909 0.0909 0.0909
 0.0909 0.0909 0.0909 0.0909 0.0909]

Es decir, hay exactamente el mismo número de elementos en cada clase, así que la muestra es válida.

## Selección de la clase de funciones a usar.

Primero vamos a ajustar un modelo lineal, ya que si obtenemos buenos resultados con él no hace falta probar con otros modelos más complejos.

Probaremos al principio sin hacer ninguna transformación de los valores observados, y si vemos que....

MODELOS: PLA, PLA_POCKET, REGRESION LINEAL, REGRESION LOGISTICA


## Identificación de las hipótesis finales que vamos a usar.

## Partición en test y entrenamiento.

Dividimos el conjunto de datos en test y entrenamiento. Para ello utilizamos la función *train_test_split*, introduciendo la aleatoriedad y reservando un 20% de los datos para test. He reservado una proporción bastante grande de los datos porque tenemos una muestra muy grande, así que los resultados que obtendremos con el 80% de los datos serán presumiblemente buenos.

Comprobamos, igual que hemos hecho antes, que los datos están bien balanceados:

Entrenamiento:
 [0.0907 0.0904 0.0913 0.0912 0.0909 0.0911
 0.0901  0.0917 0.0920 0.0907 0.09]

 [0.09177918 0.09306102 0.08921552 0.08955734 0.09109554 0.09032644
 0.09425739 0.08793369 0.08648094 0.09160827 0.09468467]
## Preprocesado de datos.

## Justificación de la métrica de error a usar.

## Justificación de los parámetros y del tipo de regularización usada.

## Selección de la mejor hipótesis del problema.

## Modelo final con todos los datos.