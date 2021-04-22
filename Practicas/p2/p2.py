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
    for i in range(x[:,0].size):
        etiqueta_real = y[i]
        etiqueta_obtenida = g(x[i,0], x[i,1])
        if (etiqueta_real != etiqueta_obtenida):
            mal_etiquetadas1+=1
            
    mal_etiquetadas2 = 0
    for i in range(x[:,0].size):
        etiqueta_real = y[i]
        etiqueta_obtenida = -g(x[i,0], x[i,1])
        if (etiqueta_real != etiqueta_obtenida):
            mal_etiquetadas2+=1
        
    mal_etiquetadas = min(mal_etiquetadas1, mal_etiquetadas2)
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
etiquetas=[]

a,b = simula_recta([-50,50]) #obtenemos los parámetros a y b de una recta aleatoria


#función auxiliar que utilizamos para calcular el porcentaje de puntos mal clasificados
def g0(x,y):
    return signo(y-a*x-b)   #MUY CUTRE VER SI SE PUEDE CAMBIAR


for i in range(0,len(x_3)): # asignamos a cada elemnto su etiqueta mediante la funcion f
    etiquetas.append(f(x_3[i,0], x_3[i,1], a,b))
    
etiquetas = np.asarray(etiquetas) #convertimos etiquetas en un arreglo

t = np.linspace(min(x_3[:,0]),max(x_3[:,0]), 100) #generamos 100 puntos entre mímino punto de la muestra y el máximo


plt.scatter(x_3[:,0], x_3[:,1], c =etiquetas) #pintamos dicha muestra, diferenciando los colores por las etiquetas
plt.plot( t, a*t+b, c = 'red') #pintamos la recta de rojo
plt.show()


print('Porcentaje mal etiquetadas:' , calculaPorcentaje(x_3,etiquetas,g0))

input("\n--- Pulsar tecla para continuar ---\n")

positivas = np.where(etiquetas == 1) #tomamos los índices de los puntos en los que la etiqueta es positiva
negativas = np.where(etiquetas == -1) #tomamos los índices de los puntos en los que la etiqueta es negativa
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

def plot_datos_cuad(X, y, fz, title='Point cloud plot', xaxis='x axis', yaxis='y axis'):
    #Preparar datos
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01
    
    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = fz(grid)
    # pred_y[(pred_y>-1) & (pred_y<1)]
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
    
    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$f(x, y)$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, 
                cmap="RdYlBu", edgecolor='white')
    
    XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]),np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]))
    positions = np.vstack([XX.ravel(), YY.ravel()])
    ax.contour(XX,YY,fz(positions.T).reshape(X.shape[0],X.shape[0]),[0], colors='black')
    
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=xaxis, ylabel=yaxis)
    plt.title(title)
    plt.show()
    
#plot_datos_cuad(x_3, etiquetas, g2)


input("\n--- Pulsar tecla para continuar ---\n")
###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 2.1: ALGORITMO PERCEPTRON

def ajusta_PLA(datos, label, max_iter, vini):
    w = 0
    
    for i in range ( 0, max_iter):
        stop = true
        for j in range (0, x.size):
            if(sign(w.T.dot(datos[j,:])) != label[j]):
                stop = false
                w = w + label[j]*x[j,:]
                
        if (stop):break
    
    return w

#CODIGO DEL ESTUDIANTE

# Random initializations
iterations = []
for i in range(0,10):
    #CODIGO DEL ESTUDIANTE
    
print('Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))

input("\n--- Pulsar tecla para continuar ---\n")

# Ahora con los datos del ejercicio 1.2.b

#CODIGO DEL ESTUDIANTE


input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT

def sgdRL(?):
    #CODIGO DEL ESTUDIANTE

    return w



#CODIGO DEL ESTUDIANTE

input("\n--- Pulsar tecla para continuar ---\n")
    


# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar Eout
# usando para ello un número suficientemente grande de nuevas muestras (>999).


#CODIGO DEL ESTUDIANTE


input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################
#BONUS: Clasificación de Dígitos


# Funcion para leer los datos
def readData(file_x, file_y, digits, labels):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la digits[0] o la digits[1]
	for i in range(0,datay.size):
		if datay[i] == digits[0] or datay[i] == digits[1]:
			if datay[i] == digits[0]:
				y.append(labels[0])
			else:
				y.append(labels[1])
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy', [4,8], [-1,1])
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy', [4,8], [-1,1])


#mostramos los datos
fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

#LINEAR REGRESSION FOR CLASSIFICATION 

#CODIGO DEL ESTUDIANTE


input("\n--- Pulsar tecla para continuar ---\n")



#POCKET ALGORITHM
  
#CODIGO DEL ESTUDIANTE




input("\n--- Pulsar tecla para continuar ---\n")


#COTA SOBRE EL ERROR

#CODIGO DEL ESTUDIANTE
