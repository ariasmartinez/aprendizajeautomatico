
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


En esta práctica vamos a realizar el ajuste y selección del mejor predictor lineal para un conjunto de datos dados. Vamos a tener dos problemas: uno de regresión y otro de clasificación, para lo que haremos dos secciones, y desarrollaremos dentro de cada sección los pasos que llevaremos a cabo, todos ellos encaminados a seleccionar el mejor modelo y la mejor estimación de error $E_{out}$

# Regresión

Para este problema utilizamos la base de datos *Superconductivty Data Data Set* encontrada en https://archive.ics.uci.edu/ml/datasets/Dataset+for+Sensorless+Drive+Diagnosis.


## Comprensión del problema a resolver. Identificación de los elementos $X, Y y f$.

El problema que queremos resolver es la asignación de una temperatura crítica a un superconductor, dadas unas características como el número de elementos, la masa atómica, el radio atómico, el punto de fusión, la entropía, etcétera. En concreto tenemos datos sobre 21263 superconductores, y de cada uno de ellos tenemos 81 características. 

Por tanto los elementos son:
$X$ : matriz con las características de los superconductores
$Y$: temperatura crítica del superconductor
$f$: función que asocia a cada superconductor su temperatura crítica.

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


## Selección de la clase de funciones a usar.


## Identificación de las hipótesis finales que vamos a usar.

## Partición en test y entrenamiento.

## Preprocesado de datos.

## Justificación de la métrica de error a usar.

## Justificación de los parámetros y del tipo de regularización usada.

## Selección de la mejor hipótesis del problema.

## Modelo final con todos los datos.