'''
PRÁCTICA 3 Clasificación 
Blanca Cano Camarero   
'''
#############################
#######  BIBLIOTECAS  #######
#############################
# Biblioteca lectura de datos
# ==========================
import pandas as pd

# matemáticas
# ==========================
import numpy as np


# Modelos a usar
# ==========================
from sklearn.linear_model import SGDClassifier



# Preprocesado 
# ==========================
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

# visualización de datos
# ==========================
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Validación cruzada
# ==========================
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut

# metricas
# ==========================
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
# Otros
# ==========================
from operator import itemgetter #ordenar lista
import time

np.random.seed(1)

###############################



########## CONSTANTES #########
NOMBRE_FICHERO_CLASIFICACION = './data/clasificacion/Sensorless_drive_diagnosis.txt'
SEPARADOR_CLASIFICACION = ' '

################### funciones auxiliares 
def LeerDatos (nombre_fichero, separador):
    '''
    Input: 
    - file_name: nombre del fichero path relativo a dónde se ejecute o absoluto
    La estructura de los datos debe ser: 
       - Cada fila un vector de características con su etiqueta en la última columna.

    Outputs: x,y
    x: matriz de filas de vector de características
    y: vector fila de la etiquetas 
    
    '''

    datos = pd.read_csv(nombre_fichero,
                       sep = separador,
                       header = None)
    valores = datos.values

    # Los datos son todas las filas de todas las columnas salvo la última 
    x = valores [:: -1]
    y = valores [:, -1] # el vector de características es la últma columana

    return x,y


def VisualizarClasificacion2D(x,y, titulo=None):
    """Representa conjunto de puntos 2D clasificados.
    Argumentos posicionales:
    - x: Coordenadas 2D de los puntos
    - y: Etiquetas"""

    _, ax = plt.subplots()
    
    # Establece límites
    xmin, xmax = np.min(x[:, 0]), np.max(x[:, 0])
    ax.set_xlim(xmin - 1, xmax + 1)
    ax.set_ylim(np.min(x[:, 1]) - 1, np.max(x[:, 1]) + 1)

    # Pinta puntos
    ax.scatter(x[:, 0], x[:, 1], c=y, cmap="tab10", alpha=0.8)

    # Pinta etiquetas
    etiquetas = np.unique(y)
    for etiqueta in etiquetas:
        centroid = np.mean(x[y == etiqueta], axis=0)
        ax.annotate(int(etiqueta),
                    centroid,
                    size=14,
                    weight="bold",
                    color="white",
                    backgroundcolor="black")

    # Muestra título
    if titulo is not None:
        plt.title(titulo)
    plt.show()


def Separador(mensaje = None):
    '''
    Hace parada del código y muestra un menaje en tal caso 
    '''
    print('\n-------- fin apartado, enter para continuar -------\n')

    if mensaje:
        print('\n' + mensaje)


###########################################################
#### Herramientas básicas


def ExploracionInicial(x):

    media = x.mean(axis = 0)
    varianza = x.var(axis = 0)
    
    print('Exploración inicial datos: \n')
    
    print('\nMedia de cada variable')
    print(media)


    print('\nVarianza ')
    print(varianza)


    print('-'*20)
    print('Resumen de las tablas')
    print('-'*20)
    
    print('\nMedia')
   
    print(f'Valor mínimo de las medias {min(media)}')
    print(f'Valor máximo de las medias {max(media)}')
    print('\nVarianza ')
   
    print(f'Valor mínimo de las varianzas {min(varianza)}')
    print(f'Valor máximo de las varianzas {max(varianza)}')
    
    print('-'*20)

    

        
###########################################################
###########################################################
###########################################################
print(f'Procedemos a leer los datos del fichero {NOMBRE_FICHERO_CLASIFICACION}')
x,y = LeerDatos( NOMBRE_FICHERO_CLASIFICACION, SEPARADOR_CLASIFICACION)

ExploracionInicial(x)

''' # COMENTO PORQUE TARDA MUCHO LA EJECUCIÓN   
print('PCA con escalado de datos')
pca_pipe = make_pipeline(StandardScaler(), PCA())
pca_pipe.fit(x)

# Se extrae el modelo entrenado del pipeline
modelo_pca = pca_pipe.named_steps['pca']
''
print('Vamos a representar los datos usando el algoritmo TSNE, este tarda un par de minutos')
x_tsne = TSNE(n_components=2).fit_transform(modelo_pca.components._modelo_pca.components_T)



tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(x)
print('t-SNE done! Time elapsed: ')

VisualizarClasificacion2D(tsne_results, y)
Separador('fin de la visualización')
'''

### Comprobación si los datos están balanceados   
def NumeroDeEtiquetas(y):
    '''
    INPUT: y: etiquetas 
    OUTPUT conteo: diccionario que asocia cada etiqueta con el número de veces que aparece 
    '''
    conteo = dict()
    etiquetas_unicas = np.unique(y)
    
    for i in etiquetas_unicas:
        conteo [i] = np.sum( y == i)
    return conteo

def ImprimeDiccionario(diccionario, titulos):

    print( ' | '.join(titulos) + '  \t  ')
    print ('--- | ' * (len(titulos)-1) + '---    ')
    for k,v in diccionario.items():
        print(k , ' | ', v , '    ')
        

print('Comprobación de balanceo')
ImprimeDiccionario(
    NumeroDeEtiquetas(y),
    ['Etiqueta', 'Número apariciones'])

Separador('Separamos test y entrenamiento')

###### separación test y entrenamiento  #####
ratio_test_size = 0.2
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size= ratio_test_size,
    shuffle = True, 
    random_state=1)

print('Veamos si ha sido homogéneo')

ImprimeDiccionario(
    NumeroDeEtiquetas(y_test),
    ['Etiqueta', 'Número apariciones'])

Separador('Normalización')  

print('Datos sin normalizar ')
ExploracionInicial(x_train)

print ('Datos normalizados')

## Normalización de los datos

scaler = StandardScaler()
x_train = scaler.fit_transform( x_train )
x_test = scaler.transform( x_test )

# No es necesario volver a comprobar la normalización
# La media deberá ser cero y la desviación típica 1, lo que no sea serán errores de redondeo.
# ExploracionInicial(x_train)

Separador('Correlación')
#------- correlacion ----

def Pearson( x, umbral, traza = False):
    '''INPUT 
    x vector de caracteríscas 
    umbral: valor mínimo del coefiente para ser tenido en cuenta
    traza: Imprime coeficiente de Pearson e índices que guardan esa relación.   

    OUTPUT
    indice_explicativo: índice de columnas linealente independientes (con ese coeficiente)
    relaciones: lista de tuplas (correlacion, índice 1, índice 2)

    '''
    r = np.corrcoef(x.T)
    longitud_r  = len(r[0])
    # Restamos la matriz identidad con la diagonal
    # Ya que queremos encontrar donde alcanza buenos niveles de correlación no triviales 
    sin_diagonal = r - np.identity(len(r[0])) 
    relaciones = [] # guardaremso tupla y 


    # Devolveré un vector con lo índices que no puedan ser explicado,
    # Esto es, si existe una correlación mayor que el umbra entre i,j
    # y I es el conjunto de características el nuevo I = I -{j}
    # Denotarelos a I con la variable índice explicativo 
    indice_explicativo = np.arange( len(x[0]))


    # muestra tupla con el coefiente de pearson y los dos índices con ese vector de características
    for i in range(longitud_r):
        for j in range(i+1, longitud_r):
            if abs(sin_diagonal[i,j]) > umbral:
            
                relaciones.append((sin_diagonal[i,j], i,j))
                #print(sin_diagonal[i,j], i,j)

                indice_explicativo [j] = 0 # El 0 siempre explicará, ya que va de mayor a menor

    indice_explicativo = np.unique(indice_explicativo)

    
    relaciones.sort(reverse=True, key =itemgetter(0))

    # imprimimos las relaciones en orden
    if(traza):
        print(f'\nCoeficiente pearson para umbral {umbral}')
        print('Coeficiente | Índice 1 | Índice 2    ')
        print( '--- | --- | ---    ')
        for i,j,k in relaciones:
            print(i,' | ' , j, ' | ', k , '    ')

    return indice_explicativo, relaciones

Separador('Índice de las características a mantener')
#print(indice_explicativo)


### Cálculos para distinto umbrales
umbrales = [0.9999, 0.999, 0.95, 0.9]
indice_explicativo = dict()
relaciones = dict()

for umbral in umbrales:
    indice_explicativo[umbral], relaciones[umbral] = Pearson( x_train,
                                                              umbral,
                                                              traza = True)
numero_caracteristicas = len(x_train[0])
print(f'\nEl número inical de características es de { numero_caracteristicas}\n' )
print('Las reducciones de dimensión total son: \n')
print('| umbral | tamaño tras reducción | reducción total |    ')
print('|:------:|:---------------------:|:---------------:|    ')
for  umbral, ie in indice_explicativo.items():
    len_ie = len(ie)
    print(f'| {umbral} | {len_ie} | {numero_caracteristicas - len_ie} |    ')


### Validación cruzada


def Evaluacion( clasificador, x, y, x_test, y_test, k_folds, nombre_modelo):
    '''
    Función para automatizar el proceso de experimento: 
    1. Ajustar modelo.
    2. Aplicar validación cruzada.
    3. Medir tiempo empleado en ajuste y validación cruzada.
    4. Medir la precisión.   

    INPUT:
    - Clasificador: Modelo con el que buscar el clasificador
    - X datos entrenamiento. 
    - Y etiquetas de los datos de entrenamiento
    - x_test, y_test
    - k-folds: número de particiones para la validación cruzada

    OUTPUT:
    '''

    ###### constantes a ajustar
    numero_trabajos_paralelos_en_validacion_cruzada = 2 
    ##########################
    
    print('\n','-'*20)
    print (f' Evaluando {nombre_modelo}')
    print('-'*20)

    
    print('\n------ Ajustando modelo------\n')        
    tiempo_inicio_ajuste = time.time()
    
    #ajustamos modelo 
    clasificador.fit(x,y) 
    tiempo_fin_ajuste = time.time()

    tiempo_ajuste = tiempo_fin_ajuste - tiempo_inicio_ajuste
    print(f'Tiempo empleado para el ajuste: {tiempo_ajuste}s')

    #validación cruzada
    tiempo_inicio_validacion_cruzada = time.time()
    resultado_validacion_cruzada = cross_val_score(
        clasificador,
        x, y,
        scoring = 'accuracy',
        cv = k_folds,
        n_jobs = numero_trabajos_paralelos_en_validacion_cruzada
    )

    tiempo_fin_validacion_cruzada = time.time()
    tiempo_validacion_cruzada = (tiempo_fin_validacion_cruzada
                                 - tiempo_inicio_validacion_cruzada)

    print(f'Tiempo empleado para validación cruzada: {tiempo_validacion_cruzada}s')

    print('Evaluación media de aciertos usando cross-validation: ',
        resultado_validacion_cruzada.mean())
    print('E_in usando cross-validation: ',
          resultado_validacion_cruzada.mean())

    # Precisión
    # predecimos test acorde al modelo
    y_predecida_test = clasificador.predict(x_test).round()
    # miramos la tasa de aciertos, es decir, cuantos ha clasificado bien
    print("\tObteniendo E_test a partir de la predicción")
    numero_aciertos = accuracy_score(y_test, y_predecida_test)
    print("\tPorcentaje de aciertos en test: ", numero_aciertos)
    print("\tE_test: ", 1 - numero_aciertos)


    




############################################################
############ EVALUACIÓN DE LOS MODELOS #####################
############################################################


k_folds = 10 # valor debe de estar entre 5 y 10


SGD = SGDClassifier(loss='hinge', penalty='l2', alpha=0.01, max_iter=5000)
Evaluacion(SGD,
        x_train, y_train,
        x_test, y_test,
        k_folds,
        'SGD Classifier con tasa de aprendizaje optima (variable) y factor de regularización 0.01 y función de perdida hinge'
        )
    



            

