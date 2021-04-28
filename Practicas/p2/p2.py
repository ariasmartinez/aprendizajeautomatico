# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: Celia Arias Martínez
"""
import numpy as np
import matplotlib.pyplot as plt


# Fijamos la semilla
np.random.seed(1)


def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gaus(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0])) 
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
    
    return out


def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b



# EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida correspondiente
#CODIGO DEL ESTUDIANTE
#obtenemos una muestra de 50 puntos en dimensión 2, en un intervalo de [-50,50] y según una distribución uniforme
print('Muestra generada con una distribución uniforme')
x_1 = simula_unif(50, 2, [-50,50]) 
plt.scatter(x_1[:,0], x_1[:,1], c ='r') #pintamos dicha muestra con el color rojo
plt.show() #mostramos la nube de puntos

input("\n--- Pulsar tecla para continuar ---\n")
#CODIGO DEL ESTUDIANTE
#obtenemos una muestra de 50 puntos en dimensión 2 según una distribución normal de media 0 y varianza
#igual a 5 en el eje de la x, y 7 en el eje de la y
print('Muestra generada con una distribución de Gauss')
x_2 = simula_gaus(50, 2, np.array([5,7])) 
plt.scatter(x_2[:,0], x_2[:,1], c ='r') #pintamos dicha muestra con el color rojo
plt.show() #mostramos la nube de puntos


###############################################################################
###############################################################################
###############################################################################


# EJERCICIO 1.2: Dibujar una gráfica con la nube de puntos de salida correspondiente

# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):
	if x >= 0:
		return 1
	return -1

def f(x, y, a, b):
	return signo(y - a*x - b)

"""
calculaPorcentaje: calcula la proporción de puntos mal clasificados
        x: muestra de puntos
        y: vector con las etiquetas
        g: función que clasifica los puntos de la muestra
        
        porcentaje_mal: proporción de puntos mal clasificada
"""
def calculaPorcentaje(x, y, g):
    
    mal_etiquetadas1 = 0
    for i in range(0,len(x[:,0])):
        etiqueta_real = y[i]
        etiqueta_obtenida = g(x[i,0], x[i,1])
        if (etiqueta_real != etiqueta_obtenida):
            mal_etiquetadas1+=1
         
    mal_etiquetadas = mal_etiquetadas1 #CAMBIAR
    #mal_etiquetadas2 = 0
    #for i in range(x[:,0].size):
     #   etiqueta_real = y[i]
      #  etiqueta_obtenida = -g(x[i,0], x[i,1])
       # if (etiqueta_real != etiqueta_obtenida):
        #    mal_etiquetadas2+=1
        
    #mal_etiquetadas = min(mal_etiquetadas1, mal_etiquetadas2)
    porcentaje_mal = mal_etiquetadas / x[:,0].size
    
    return porcentaje_mal

#CODIGO DEL ESTUDIANTE


input("\n--- Pulsar tecla para continuar ---\n")

# 1.2.b. Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto con la recta usada para ello
# Array con 10% de indices aleatorios para introducir ruido

#CODIGO DEL ESTUDIANTE


print('Ejercicio 1.2: muestra con distribución uniforme')
#muestra de puntos con distribución uniforme
x_3 = simula_unif(100, 2, [-50,50])
etiquetas_originales=[]

a,b = simula_recta([-50,50]) #obtenemos los parámetros a y b de una recta aleatoria


#función auxiliar que utilizamos para calcular el porcentaje de puntos mal clasificados
def g0(x,y):
    return signo(y-a*x-b)   #MUY CUTRE VER SI SE PUEDE CAMBIAR


for i in range(0,len(x_3)): # asignamos a cada elemnto su etiqueta mediante la funcion f
    etiquetas_originales.append(f(x_3[i,0], x_3[i,1], a,b))
    
etiquetas_originales = np.asarray(etiquetas_originales) #convertimos etiquetas en un arreglo

t = np.linspace(min(x_3[:,0]),max(x_3[:,0]), 100) #generamos 100 puntos entre mímino punto de la muestra y el máximo


plt.scatter(x_3[:,0], x_3[:,1], c =etiquetas_originales) #pintamos dicha muestra, diferenciando los colores por las etiquetas
plt.plot( t, a*t+b, c = 'red') #pintamos la recta de rojo
plt.show()


print('Porcentaje mal etiquetadas:' , calculaPorcentaje(x_3,etiquetas_originales,g0))

input("\n--- Pulsar tecla para continuar ---\n")

positivas = np.where(etiquetas_originales == 1) #tomamos los índices de los puntos en los que la etiqueta es positiva
negativas = np.where(etiquetas_originales == -1) #tomamos los índices de los puntos en los que la etiqueta es negativa
positivas = np.asarray(positivas).T #trasponemos los dos vectores
negativas = np.asarray(negativas).T

#tomamos de forma aleatoria un 10% del total de las positivas
ind_pos = np.random.choice(len(positivas), int(0.1*len(positivas)), replace = True)
#del vector de positivos nos quedamos con los valores de los índices obtenidos, por 
#lo tanto nos estamos quedando con un 10% de los índices de los valores positivos del vector original
cambiar_signo = positivas[ind_pos,:]
#hacemos lo mismo con el vector de índices de valores negativos
ind_neg = np.random.choice(len(negativas), int(0.1*len(negativas)), replace = True)
#concatenamos los dos vectores de índices obtenidos 
cambiar_signo = np.concatenate((cambiar_signo, negativas[ind_neg,:]), axis=0)

#cambiamos los valores de las etiquetas de los índices obtenidos
etiquetas = np.copy(etiquetas_originales)
for i in range(0, len(cambiar_signo)):
    etiquetas[cambiar_signo[i]]=-etiquetas[cambiar_signo[i]] 
    
#tomamos 100 valores espaciados entre el mínimo y el máximo de los valores
t = np.linspace(min(x_3[:,0]),max(x_3[:,0]), 100)
plt.scatter(x_3[:,0], x_3[:,1], c =etiquetas) #pintamos dicha muestra
plt.plot( t, a*t+b, c = 'red') #pintamos la recta 
plt.show()

#calculamos el porcentaje de mal etiquetadas, le pasamos la función g0 que es la recta
#con los parámetros fijos
print('Porcentaje mal etiquetadas:' , calculaPorcentaje(x_3,etiquetas,g0))

input("\n--- Pulsar tecla para continuar ---\n")
###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 1.3: Supongamos ahora que las siguientes funciones definen la frontera de clasificación de los puntos de la muestra en lugar de una recta

def g1(x,y):
    return signo((x-10)**2+(y-20)**2-400)

def g2(x,y):
    return signo(0.5*(x+10)**2+(y-20)**2-400)

def g3(x,y):
    return signo((0.5*(x-10)**2-(y+20)**2-400))

def g4(x,y):
    return signo((y-20*x**2-5*x+3))

print( 'f(x,y) = (x-10)^2+(y-20)^2-400')


t = np.linspace(min(x_3[:,0]),max(x_3[:,0]), 100)
#me quedo con las posiciones en las que se cumple la desigualdad que nos dice que el valor de x pertenece al dominio
posiciones_dominio = np.where((-10 <= t))
t = t[posiciones_dominio] #genero un vector con los valores de x que están en el dominio

posiciones_dominio = np.where(t <= 30)
t = t[posiciones_dominio] #genero un vector con los valores de x que están en el dominio



plt.scatter(x_3[:,0], x_3[:,1], c =etiquetas) #pintamos dicha muestra
plt.plot( t, np.sqrt(-t**2+20*t+300)+20, c = 'red') #pintamos las dos soluciones de la ecuación
plt.plot(t, 20-np.sqrt(-t**2+20*t+300), c = 'red') 
plt.show()

print('Porcentaje mal etiquetadas:' , calculaPorcentaje(x_3,etiquetas,g1))

input("\n--- Pulsar tecla para continuar ---\n")
print('f(x,y) = 0.5*(x+10)^2+(y-20)^2-400')




t = np.linspace(min(x_3[:,0]),max(x_3[:,0]), 100)

#me quedo con las posiciones en las que se cumple la desigualdad que nos dice que el valor de x pertenece al dominio
posiciones_dominio = np.where(((-10-20*np.sqrt(2)) <= t))
t = t[posiciones_dominio] #genero un vector con los valores de x que están en el dominio

posiciones_dominio = np.where(t <= (20*np.sqrt(2)-10))
t = t[posiciones_dominio] #genero un vector con los valores de x que están en el dominio



plt.scatter(x_3[:,0], x_3[:,1], c =etiquetas) #pintamos dicha muestra
plt.plot( t, 1/2*(40-np.sqrt(2)*np.sqrt(-t**2-20*t+700)), c = 'red') #pintamos las dos soluciones de la ecuación

plt.plot(t, 1/2*(40+np.sqrt(2)*np.sqrt(-t**2-20*t+700)), c = 'red')
plt.show()

print('Porcentaje mal etiquetadas:' , calculaPorcentaje(x_3,etiquetas,g2))

input("\n--- Pulsar tecla para continuar ---\n")

print('f(x,y) = 0.5*(x-10)^2-(y+20)^2-40' )




t = np.linspace(min(x_3[:,0]),max(x_3[:,0]), 100)

#me quedo con las posiciones en las que se cumple la desigualdad que nos dice que el valor de x pertenece al dominio
posiciones_dominio1 = np.where((10+20*np.sqrt(2)) <= t)

posiciones_dominio2 = np.where(t <= (-20*np.sqrt(2)+10))


posiciones_dominio = np.union1d(posiciones_dominio1, posiciones_dominio2)
t = t[posiciones_dominio] #genero un vector con los valores de x que están en el dominio



plt.scatter(x_3[:,0], x_3[:,1], c =etiquetas) #pintamos dicha muestra
plt.plot( t, 1/2*(-40-np.sqrt(2)*np.sqrt(t**2-20*t-700)), c = 'red') #pintamos las dos soluciones de la ecuación
plt.plot(t,  1/2*(-40+np.sqrt(2)*np.sqrt(t**2-20*t-700)), c = 'red')
plt.show()


print('Porcentaje mal etiquetadas:' , calculaPorcentaje(x_3,etiquetas,g3))


input("\n--- Pulsar tecla para continuar ---\n")

print('f(x,y) = y-20*x^2-5*x+3')



t = np.linspace(min(x_3[:,0]),max(x_3[:,0]), 1000000) #necesito más puntos, ya que los que finalmente utilizaré pertenecen a un intervalo muy pequeño
posiciones_dominio = np.where(20*t**2+5*t-3 < max(x_3[:,1])) #tomo los puntos que no se salen del recinto

t = t[posiciones_dominio]
plt.scatter(x_3[:,0], x_3[:,1], c =etiquetas) #pintamos dicha muestra
plt.plot( t,20*t**2+5*t-3, c = 'red')

plt.show()


print('Porcentaje mal etiquetadas:' , calculaPorcentaje(x_3,etiquetas,g4))


input("\n--- Pulsar tecla para continuar ---\n")
###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 2.1: ALGORITMO PERCEPTRON

def ajusta_PLA(datos, label, max_iter, vini):
    w = np.copy(vini)
    iteraciones = 0
    for i in range ( 0, max_iter):
        iteraciones+=1
        stop = True
        for j in range (0, len(datos)):
            if(signo(w.T.dot(datos[j,:]).reshape(-1,1)) != label[j]):
                stop = False
                w = w + label[j]*datos[j,:].reshape(-1,1)
                
        if (stop):break
    
    return w, iteraciones


def aux_1(x,y):  # w0+ w1*x1+ w1*x2 = 0
    return signo(w[0]+w[1]*x+w[1]*y)
#CODIGO DEL ESTUDIANTE

#ejecutamos la función ajusta_PLA con los datos del apartado 2a y como vector inicial
#el vector 0
vector_unos = np.ones((len(x_3),1))
datos = np.copy(x_3)
datos = np.concatenate((vector_unos, datos), axis = 1)
    
vector_inicial = np.zeros((datos[0].size,1)).reshape(-1,1)

w, it = ajusta_PLA(datos, etiquetas_originales, 100, vector_inicial)
print('w: ', w)
print('Número de iteraciones: ', it)
print('Porcentaje mal etiquetadas:' , calculaPorcentaje(x_3,etiquetas_originales,aux_1))

plt.scatter(x_3[:,0], x_3[:,1], c =etiquetas_originales)
t = np.linspace(min(x_3[:,0]),max(x_3[:,1]), 100)
plt.plot( t, (-w[0]-w[1]*t)/w[2], c = 'red')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

iterations = []
porcentajes_mal_etiquetadas = []

for i in range(0,10):
    aleat = np.random.uniform(0,1,(datos[0].size, 1)).reshape(-1,1)
    w, it = ajusta_PLA(datos, etiquetas_originales, 100, aleat)
    porcentajes_mal_etiquetadas.append(calculaPorcentaje(x_3,etiquetas_originales,aux_1))
    iterations.append(it)
    plt.scatter(x_3[:,0], x_3[:,1], c =etiquetas_originales)
    t = np.linspace(min(x_3[:,0]),max(x_3[:,1]), 100)
    plt.plot( t, (-w[0]-w[1]*t)/w[2], c = 'red')
    plt.show()
    


iterations = np.asarray(iterations)
porcentajes_mal_etiquetadas = np.asarray(porcentajes_mal_etiquetadas)
print('Porcentajes mal etiquedadas: ', porcentajes_mal_etiquetadas)
print('Iteraciones con cada valor inicial: ', iterations)
print('Número medio de iteraciones para converger: ', iterations.mean())
print('Porcentaje medio de mal etiquetadas: ', porcentajes_mal_etiquetadas.mean())

input("\n--- Pulsar tecla para continuar ---\n")

#Lo hacemos ahora con una muestra de puntos con ruido
vector_unos = np.ones((len(x_3),1))
datos = np.copy(x_3)
datos = np.concatenate((vector_unos, datos), axis = 1)
    
vector_inicial = np.zeros((datos[0].size,1)).reshape(-1,1)

w, it = ajusta_PLA(datos, etiquetas, 100, vector_inicial)
print('w: ', w)
print('Número de iteraciones: ', it)


    
print('Porcentaje mal etiquetadas:' , calculaPorcentaje(x_3,etiquetas,aux_1))  #CORREGIR

plt.scatter(x_3[:,0], x_3[:,1], c =etiquetas)
t = np.linspace(min(x_3[:,0]),max(x_3[:,1]), 100)
plt.plot( t, (-w[0]-w[1]*t)/w[2], c = 'red')
plt.show()





input("\n--- Pulsar tecla para continuar ---\n")

iterations = []

porcentajes_mal_etiquetadas = []

suma_it = 0
for i in range(0,10):
    aleat = np.random.uniform(0,1,(datos[0].size, 1)).reshape(-1,1)
    w, it = ajusta_PLA(datos, etiquetas, 100, aleat)
    #suma_it+=it
    porcentajes_mal_etiquetadas.append(calculaPorcentaje(x_3,etiquetas,aux_1))
    iterations.append(it)
    #print('Porcentaje mal etiquetadas:' , calculaPorcentaje(x_3,etiquetas,aux_1))
    plt.scatter(x_3[:,0], x_3[:,1], c =etiquetas)
    t = np.linspace(min(x_3[:,0]),max(x_3[:,1]), 100)
    plt.plot( t, (-w[0]-w[1]*t)/w[2], c = 'red')
    plt.show()



iterations = np.asarray(iterations)
porcentajes_mal_etiquetadas = np.asarray(porcentajes_mal_etiquetadas)
print('Porcentajes mal etiquedadas: ', porcentajes_mal_etiquetadas)
print('Iteraciones con cada valor inicial: ', iterations)
print('Número medio de iteraciones para converger: ', iterations.mean())
print('Porcentaje medio de mal etiquetadas: ', porcentajes_mal_etiquetadas.mean())

input("\n--- Pulsar tecla para continuar ---\n")
    

#REGRESIÓN LOGÍSTICA

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT

""" N = 1 """
def sgdRL(x, y, max_iter, epsilon ,eta):
    w = np.zeros((x[0].size,1)).T
    y = y.reshape((1, -1))  #convertimos y en un vector columna para poder realizar la multiplicación
    for i in range (max_iter):
        w_old = np.copy(w)
        grad = -(y.dot(x))/(1+np.exp((y.dot((w.dot(x.T)).T))))
        w = w_old -eta*grad
        if ((w-w_old).max(axis = 1) < epsilon): break

    return w.T



x_entre = simula_unif(500, 2, [0,2]) 
indice = np.random.randint(0,50, (2,1))
punto_a =[ x_entre[indice[0],0], x_entre[indice[0],1]]
punto_b =[ x_entre[indice[1],0], x_entre[indice[1],1]]

a = (punto_b[1]-punto_b[0])/(punto_a[1]-punto_b[0]) # Calculo de la pendiente.
b = punto_b[0] - a*punto_a[0]       # Calculo del termino independiente.

def h(x,y):
    return signo(y-a*x-b)

etiquetas = []
for i in range(0,len(x_entre)): # asignamos a cada elemnto su etiqueta mediante la funcion f
    etiquetas.append(h(x_entre[i,0], x_entre[i,1]))

    
etiquetas = np.asarray(etiquetas) #convertimos etiquetas en un arreglo

t = np.linspace(min(x_entre[:,0]),max(x_entre[:,0]), 100) #generamos 100 puntos entre mímino punto de la muestra y el máximo
plt.scatter(x_entre[:,0], x_entre[:,1], c =etiquetas) #pintamos dicha muestra, diferenciando los colores por las etiquetas
plt.plot( t, a*t+b, c = 'red') #pintamos la recta de rojo
plt.show()


input("\n--- Pulsar tecla para continuar ---\n")
    
    
N = 100

vector_unos = np.ones((len(x_entre),1))
datos_entre = np.copy(x_entre)
datos_entre = np.concatenate((vector_unos, datos_entre), axis = 1)
w = sgdRL(datos_entre, etiquetas, 100000, 0.01, 0.05)
 
   
t = np.linspace(min(x_entre[:,0]),max(x_entre[:,0]), 100) #generamos 100 puntos entre mímino punto de la muestra y el máximo
plt.scatter(x_entre[:,0], x_entre[:,1], c =etiquetas) #pintamos dicha muestra, diferenciando los colores por las etiquetas
plt.plot( t, (-w[0]-w[1]*t)/w[2], c = 'red') #pintamos la recta de rojo
plt.show()




