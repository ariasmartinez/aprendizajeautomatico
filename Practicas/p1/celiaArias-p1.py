#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Nombre Estudiante:  Celia Arias Martínez
"""

import numpy as np
import matplotlib.pyplot as plt
import math 
from sklearn.utils import shuffle

np.random.seed(1)

print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')
print('Ejercicio 1\n')

def E(u,v):
    return (u**3*np.e**(v-2)-2*v**2*np.e**(-u))**2

#Derivada parcial de E con respecto a u
def dEu(u,v):
    return 2*(u**3*np.e**(v-2)-2*v**2*np.e**(-u))*(3*u**2*np.e**(v-2)+np.e**(-u)*2*v**2)
    
#Derivada parcial de E con respecto a v
def dEv(u,v):
    return 2*(u**3*np.e**(v-2)-2*v**2*np.e**(-u))*(u**3*np.e**(v-2)-4*v*np.e**(-u))

#Gradiente de E
def gradE(u,v):
    return np.array([dEu(u,v), dEv(u,v)])


"""
gradient_descent: función que implementa el algoritmo del gradiente descendiente
    p: punto inicial
    lr: learning rate
    maxIter: máximo de iteraciónes
    funcion: función a minimizar
    gradiente: gradiente de la funcion
    error_fijado : cota de error
    
    w: punto final donde se alcanza el mínimo obtenido
    iterations: nº iteraciones realizado
"""
def gradient_descent(p, lr, maxIter, funcion, gradiente, error_fijado ):
    iterations = 0
    w = [p[0], p[1]]
    for j in range(maxIter):
        iterations += 1
        w = w - lr*gradiente(w[0], w[1])
        if (funcion(w[0], w[1]) < error_fijado) : break
    return w, iterations    


eta = 0.1 
maxIter = 10000000000
error2get = 1e-14
initial_point = np.array([1.0,1.0])
w, it = gradient_descent(initial_point, eta, maxIter, E, gradE, error2get)

print('Gradiente de E: 2*(u**3*np.e**(v-2)-2*v**2*np.e**(-u))*(3*u**2*np.e**(v-2)+np.e**(-u)*2*v**2), 2*(u**3*np.e**(v-2)-2*v**2*np.e**(-u))*(u**3*np.e**(v-2)-4*v*np.e**(-u))')
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')



input("\n--- Pulsar tecla para continuar ---\n")
# Ejercicio 1.3-a

"""
gradient_descent_errores: función que implementa el algoritmo de gradiente descendente y almacena los sucesivos valores de los puntos encontrados
    p : punto inicial
    lr: learning rate
    maxIter: máximo de iteraciones a realizar
    funcion: función que queremos minimizar
    gradiente: gradiente de dicha función
    
    w: punto final donde se alcanza el mínimo obtenido
    iterations: nº iteraciones realizado
    errores: vector con los errores que obtenemos para los diferentes valores de w
"""

def gradient_descent_errores(p, lr, maxIter, funcion, gradiente):
    iterations = 0
    w = [p[0], p[1]]
    errores = np.zeros(51, dtype=float)
    errores[iterations] = funcion(w[0], w[1])
    
    for j in range(maxIter):
        iterations += 1
        w = w - lr*gradiente(w[0], w[1])
        errores[iterations] = funcion(w[0], w[1])
   
    return w, iterations, errores

def f(u,v):
    return (u+2)**2+2*(v-2)**2+2*np.sin(2*math.pi*u)*np.sin(2*math.pi*v)

#Derivada parcial de E con respecto a u
def dfu(u,v):
    return 2*(u+2)+4*math.pi*np.sin(2*math.pi*v)*np.cos(2*math.pi*u)
    
#Derivada parcial de E con respecto a v
def dfv(u,v):
    return 4*(v-2)+4*math.pi*np.sin(2*math.pi*u)*np.cos(2*math.pi*v)

#Gradiente de E
def gradf(u,v):
    return np.array([dfu(u,v), dfv(u,v)])



initial_point2 = np.array([-1.0,1.0])
eta = 0.01
maxIter2 = 50
punto_final, it, diferencias = gradient_descent_errores(initial_point2, eta, maxIter2, f, gradf)
print ('Valor de la función para lr de 0.01: ', f(punto_final[0], punto_final[1]))
print ('Coordenadas obtenidas para lr de 0.01: (', punto_final[0], ', ', punto_final[1],')')

eta = 0.1
w3, it3, diferencias2 = gradient_descent_errores(initial_point2, eta, maxIter2, f, gradf)
print ('Valor de la función para lr de 0.1: ', f(punto_final[0], punto_final[1]))
print ('Coordenadas obtenidas para lr de 0.1: (', w3[0], ', ', w3[1],')')

input("\n--- Pulsar tecla para continuar ---\n")

eta = 0.005
punto_final, it, diferencias3 = gradient_descent_errores(initial_point2, eta, maxIter2, f, gradf)
print ('Valor de la función para lr de 0.001: ', f(punto_final[0], punto_final[1]))
print ('Coordenadas obtenidas para lr de 0.001: (', punto_final[0], ', ', punto_final[1],')')


eta = 0.05
punto_final, it, diferencias4 = gradient_descent_errores(initial_point2, eta, maxIter2, f, gradf)
print ('Valor de la función para lr de 0.05: ', f(punto_final[0], punto_final[1]))
print ('Coordenadas obtenidas para lr de 0.05 (', punto_final[0], ', ', punto_final[1],')')

input("\n--- Pulsar tecla para continuar ---\n")

fig = plt.figure()
#poner que automatico el eta
plt.plot(range(0,51), diferencias, label = 'eta: 0.01')
plt.plot(range(0,51), diferencias2, label = 'eta: 0.1')
plt.legend()
plt.xlabel('Iteraciones')
plt.ylabel('Valor de la funcion' )
plt.title('Ejercicio 1.3')
plt.show()


fig = plt.figure()
plt.plot(range(0,51), diferencias, label = 'eta: 0.01')
plt.plot(range(0,51), diferencias2, label = 'eta: 0.1')
plt.plot(range(0,51), diferencias3, label = 'eta: 0.005')
plt.plot(range(0,51), diferencias4, label = 'eta: 0.05')
plt.legend()
plt.xlabel('Iteraciones')
plt.ylabel('Valor de la funcion' )
plt.title('Ejercicio 1.3')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")


#Ejercicio 3.b eta 0.01
print('Tasa de aprendizaje : 0.01')
eta = 0.01
punto_final, it, diferencias_a = gradient_descent_errores(np.array([-0.5,-0.5]), eta, maxIter2, f, gradf)
print ('Punto inicial: [-0.5,-0.5]')
print ('Coordenadas obtenidas: (', punto_final[0], ', ', punto_final[1],')')
print ('Valor de la funcion: ' , diferencias_a[50])

punto_final, it, diferencias_b = gradient_descent_errores(np.array([1,1]), eta, maxIter2, f, gradf)
print ('Punto inicial: [1,1]')
print ('Coordenadas obtenidas: (', punto_final[0], ', ', punto_final[1],')')
print ('Valor de la funcion: ' , diferencias_b[50])

punto_final, it, diferencias_c = gradient_descent_errores(np.array([2.1,-2.1]), eta, maxIter2, f, gradf)
print ('Punto inicial: [2.1,-2.1]')
print ('Coordenadas obtenidas: (', punto_final[0], ', ', punto_final[1],')')
print ('Valor de la funcion: ' , diferencias_c[50])


punto_final, it, diferencias_e = gradient_descent_errores(np.array([-3,3]), eta, maxIter2, f, gradf)
print ('Punto inicial: [-3,3]')
print ('Coordenadas obtenidas: (', punto_final[0], ', ', punto_final[1],')')
print ('Valor de la funcion: ' , diferencias_e[50])

punto_final, it, diferencias_f = gradient_descent_errores(np.array([-2,2]), eta, maxIter2, f, gradf)
print ('Punto inicial: [-2,2]')
print ('Coordenadas obtenidas: (', punto_final[0], ', ', punto_final[1],')')
print ('Valor de la funcion: ' , diferencias_f[50])

input("\n--- Pulsar tecla para continuar ---\n")


fig = plt.figure()

plt.plot(range(0,51), diferencias_a, label = '[-0.5,-0.5]')
plt.plot(range(0,51), diferencias_b, label = '[1,1]')
plt.plot(range(0,51), diferencias_c, label = '[2.1,-2.1]')

plt.plot(range(0,51), diferencias_e, label = '[-3,3]')
plt.plot(range(0,51), diferencias_f, label = '[-2,2]')
plt.legend()
plt.xlabel('Iteraciones')
plt.ylabel('Valor de la funcion' )
plt.title('Ejercicio 1.3-b')
plt.show()


input("\n--- Pulsar tecla para continuar ---\n")
#Ejercicio 3-b eta 0.1
eta = 0.1
print('Tasa de aprendizaje igual a 0.1')
punto_final, it, diferencias_a = gradient_descent_errores(np.array([-0.5,-0.5]), eta, maxIter2, f, gradf)
print ('Punto inicial: [-0.5,-0.5]')
print ('Coordenadas obtenidas: (', punto_final[0], ', ', punto_final[1],')')
print ('Valor de la funcion: ' , diferencias_a[50])

punto_final, it, diferencias_b = gradient_descent_errores(np.array([1,1]), eta, maxIter2, f, gradf)
print ('Punto inicial: [1,1]')
print ('Coordenadas obtenidas: (', punto_final[0], ', ', punto_final[1],')')
print ('Valor de la funcion: ' , diferencias_b[50])

punto_final, it, diferencias_c = gradient_descent_errores(np.array([2.1,-2.1]), eta, maxIter2, f, gradf)
print ('Punto inicial: [2.1,-2.1]')
print ('Coordenadas obtenidas: (', punto_final[0], ', ', punto_final[1],')')
print ('Valor de la funcion: ' , diferencias_c[50])

punto_final, it, diferencias_e = gradient_descent_errores(np.array([-3,3]), eta, maxIter2, f, gradf)
print ('Punto inicial: [-3,3]')
print ('Coordenadas obtenidas: (', punto_final[0], ', ', punto_final[1],')')
print ('Valor de la funcion: ' , diferencias_e[50])

punto_final, it, diferencias_f = gradient_descent_errores(np.array([-2,2]), eta, maxIter2, f, gradf)
print ('Punto inicial: [-2,2]')
print ('Coordenadas obtenidas: (', punto_final[0], ', ', punto_final[1],')')
print ('Valor de la funcion: ' , diferencias_f[50])

input("\n--- Pulsar tecla para continuar ---\n")


fig = plt.figure()

plt.plot(range(0,51), diferencias_a, label = '[-0.5,-0.5]')
plt.plot(range(0,51), diferencias_b, label = '[1,1]')
plt.plot(range(0,51), diferencias_c, label = '[2.1,-2.1]')
plt.plot(range(0,51), diferencias_e, label = '[-3,3]')
plt.plot(range(0,51), diferencias_f, label = '[-2,2]')
plt.legend()
plt.xlabel('Iteraciones')
plt.ylabel('Valor de la funcion' )
plt.title('Ejercicio 1.3-b')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")


#Ejercicio 3-b eta 0.05
eta = 0.05
print('Tasa de aprendizaje igual a 0.05')
punto_final, it, diferencias_a = gradient_descent_errores(np.array([-0.5,-0.5]), eta, maxIter2, f, gradf)
print ('Punto inicial: [-0.5,-0.5]')
print ('Coordenadas obtenidas: (', punto_final[0], ', ', punto_final[1],')')
print ('Valor de la funcion: ' , diferencias_a[50])

punto_final, it, diferencias_b = gradient_descent_errores(np.array([1,1]), eta, maxIter2, f, gradf)
print ('Punto inicial: [1,1]')
print ('Coordenadas obtenidas: (', punto_final[0], ', ', punto_final[1],')')
print ('Valor de la funcion: ' , diferencias_b[50])

punto_final, it, diferencias_c = gradient_descent_errores(np.array([2.1,-2.1]), eta, maxIter2, f, gradf)
print ('Punto inicial: [2.1,-2.1]')
print ('Coordenadas obtenidas: (', punto_final[0], ', ', punto_final[1],')')
print ('Valor de la funcion: ' , diferencias_c[50])

punto_final, it, diferencias_e = gradient_descent_errores(np.array([-3,3]), eta, maxIter2, f, gradf)
print ('Punto inicial: [-3,3]')
print ('Coordenadas obtenidas: (', punto_final[0], ', ', punto_final[1],')')
print ('Valor de la funcion: ' , diferencias_e[50])

punto_final, it, diferencias_f = gradient_descent_errores(np.array([-2,2]), eta, maxIter2, f, gradf)
print ('Punto inicial: [-2,2]')
print ('Coordenadas obtenidas: (', punto_final[0], ', ', punto_final[1],')')
print ('Valor de la funcion: ' , diferencias_f[50])

input("\n--- Pulsar tecla para continuar ---\n")


fig = plt.figure()

plt.plot(range(0,51), diferencias_a, label = '[-0.5,-0.5]')
plt.plot(range(0,51), diferencias_b, label = '[1,1]')
plt.plot(range(0,51), diferencias_c, label = '[2.1,-2.1]')
plt.plot(range(0,51), diferencias_e, label = '[-3,3]')
plt.plot(range(0,51), diferencias_f, label = '[-2,2]')
plt.legend()
plt.xlabel('Iteraciones')
plt.ylabel('Valor de la funcion' )
plt.title('Ejercicio 1.3-b')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")


#Ejercicio 3-b eta 0.5
eta = 0.5
print('Tasa de aprendizaje igual a 0.5')
punto_final, it, diferencias_a = gradient_descent_errores(np.array([-0.5,-0.5]), eta, maxIter2, f, gradf)
print ('Punto inicial: [-0.5,-0.5]')
print ('Coordenadas obtenidas: (', punto_final[0], ', ', punto_final[1],')')
print ('Valor de la funcion: ' , diferencias_a[50])

punto_final, it, diferencias_b = gradient_descent_errores(np.array([1,1]), eta, maxIter2, f, gradf)
print ('Punto inicial: [1,1]')
print ('Coordenadas obtenidas: (', punto_final[0], ', ', punto_final[1],')')
print ('Valor de la funcion: ' , diferencias_b[50])

punto_final, it, diferencias_c = gradient_descent_errores(np.array([2.1,-2.1]), eta, maxIter2, f, gradf)
print ('Punto inicial: [2.1,-2.1]')
print ('Coordenadas obtenidas: (', punto_final[0], ', ', punto_final[1],')')
print ('Valor de la funcion: ' , diferencias_c[50])

punto_final, it, diferencias_e = gradient_descent_errores(np.array([-3,3]), eta, maxIter2, f, gradf)
print ('Punto inicial: [-3,3]')
print ('Coordenadas obtenidas: (', punto_final[0], ', ', punto_final[1],')')
print ('Valor de la funcion: ' , diferencias_e[50])

punto_final, it, diferencias_f = gradient_descent_errores(np.array([-2,2]), eta, maxIter2, f, gradf)
print ('Punto inicial: [-2,2]')
print ('Coordenadas obtenidas: (', punto_final[0], ', ', punto_final[1],')')
print ('Valor de la funcion: ' , diferencias_f[50])

input("\n--- Pulsar tecla para continuar ---\n")


fig = plt.figure()

plt.plot(range(0,51), diferencias_a, label = '[-0.5,-0.5]')
plt.plot(range(0,51), diferencias_b, label = '[1,1]')
plt.plot(range(0,51), diferencias_c, label = '[2.1,-2.1]')
plt.plot(range(0,51), diferencias_e, label = '[-3,3]')
plt.plot(range(0,51), diferencias_f, label = '[-2,2]')
plt.legend()
plt.xlabel('Iteraciones')
plt.ylabel('Valor de la funcion' )
plt.title('Ejercicio 1.3-b')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")
#Ejercicio 2

print('EJERCICIO SOBRE REGRESION LINEAL\n')
print('Ejercicio 1\n')

label5 = 1
label1 = -1

# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(label5)
			else:
				y.append(label1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Funcion para calcular el error
"""
Err: función que calcula el error dado un vector de pesos
    x: vector de características
    y: vector de etiquetas
    w: vector de pesos
    
    return: float con el error obtenido
"""
def Err(x,y,w):
    y = y.reshape((-1,1))
    z = x.dot(w)-y
    return (1/len(y))*z.T.dot(z)  

# Gradiente Descendente original
def sgdN(x,y, maxIter, eta):
    w = np.zeros((x[0].size,1))
    iterations = 0
    y = y.reshape((-1, 1))
    for j in range(maxIter):
        iterations+=1
        w = w -(2/len(x))*eta*(x.T.dot(x.dot(w)-y)) 
    return w

#Gradiente descendiente estocastico
"""
sgd: funcion que implementa el algoritmo de gradiente descendiente estocástico
    x: vector de características
    y: vector de etiquetas
    maxIter: máximo de iteraciones
    eta: tasa de aprendizaje
    tam_mini_bach: tamaño de mini bach
    
    w: vector de pesos
"""
def sgd(x,y, maxIter, eta, tam_mini_bach):
    w = np.zeros((x[0].size,1)) #vector columna del tamaño del número de muestras, inicializado a cero
    iterations = 0 
    y = y.reshape((-1, 1))  #convertimos y en un vector columna para poder realizar la multiplicación
    x_aux = np.copy(x)  #hacemos una copia de x que será con la que trabajemos
    x_aux = np.c_[x,y]  #concatenamos x con y, es decir, ponemos las etiquetas como la última columna de x
   
    for j in range(maxIter):
        iterations+=1
        x_aux = shuffle(x_aux, random_state = 0) #mezclamos el vector conjunto de características y etiquetas
        num_mini_bach = int(len(x_aux)/tam_mini_bach) #calculamos el número de mini bach que vamos a tener
        pos = 0
   
        for i in range(0, num_mini_bach): #para cada mini bach:
            tam_prov = tam_mini_bach if ((len(x)-pos) > tam_mini_bach) else len(x)-pos #fijamos el tamaño de mini bach, que será tam_mini_bach, a no ser que estemos en el último mini bach y no queden más elementos
            bach = x_aux[pos:pos+tam_prov,:] #generamos el primer mini bach
            y_aux = bach[:,(x_aux[0].size-1)] #generamos las etiquetas correspondientes al primer mini bach
            y_aux = y_aux.reshape((-1,1)) #convertimos y en un vector columna
            bach = bach[:,0:(x_aux[0].size-1)] #eliminamos la columna correspondiente a las etiquetas
            w = w - (2/len(bach))*eta*(bach.T.dot(bach.dot(w)-y_aux)) #aplicamos gradiente descendiente al mini bach
            pos+=tam_prov #actualizamos el pivote 
    return w



# Pseudoinversa	
"""
pseudoinverse: función que aplica el algoritmo de la pseudoinversa
    x: vector con las características
    y: vector con las etiquetas
    
    return: vector de pesos
"""
def pseudoinverse(x,y):
    y = y.reshape((-1,1))
    return (np.linalg.inv(x.T.dot(x))).dot(x.T.dot(y))


# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')


w_pseudo = pseudoinverse(x,y)
print ('Bondad del resultado para pseudoinversa:\n')
print ("Ein: ", Err(x,y,w_pseudo))
print ("Eout: ", Err(x_test, y_test, w_pseudo))
print("w: ", w_pseudo)

#Dibujamos la recta obtenida
y0 = np.where(y == -1)
y1 = np.where(y == 1)

#Hacemos 3 arrays para separar los indices de las clases 0,1 y 2
x_2 = np.array([x[y0],x[y1]])

plt.scatter(x_2[0][:, 1], x_2[0][:, 2],  c = 'blue', label = '1')
plt.scatter(x_2[1][:, 1], x_2[1][:, 2],  c = 'orange', label = '5')
t = np.linspace(0,1, 100)
plt.plot( t, (-w_pseudo[0]-w_pseudo[1]*t)/w_pseudo[2], c = 'red')
plt.legend();

plt.figure()
plt.show()


input("\n--- Pulsar tecla para continuar ---\n")
w = sgd(x,y,200,0.1, 32)
print ('Bondad del resultado para grad. descendente estocastico, tamaño de mini bach 32:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))
print("w: ", w)


y0 = np.where(y == -1)
y1 = np.where(y == 1)

#Hacemos 3 arrays para separar los indices de las clases 0,1 y 2
x_2 = np.array([x[y0],x[y1]])

plt.scatter(x_2[0][:, 1], x_2[0][:, 2],  c = 'blue', label = '1')
plt.scatter(x_2[1][:, 1], x_2[1][:, 2],  c = 'orange', label = '5')
t = np.linspace(0,1, 100)
plt.plot( t, (-w[0]-w[1]*t)/w[2], c = 'red', label = "sgd tam 32")
plt.plot( t, (-w_pseudo[0]-w_pseudo[1]*t)/w_pseudo[2], c = 'green', label = "pseudoinverse")
plt.legend();
plt.title("Ejercicio1")

plt.figure()
plt.show()


input("\n--- Pulsar tecla para continuar ---\n")
w_4 = sgd(x,y,200,0.1, 17)
print ('Bondad del resultado para grad. descendente estocastico, tamaño de mini bach 7:\n')
print ("Ein: ", Err(x,y,w_4))
print ("Eout: ", Err(x_test, y_test, w_4))
print("w: ", w_4)


y0 = np.where(y == -1)
y1 = np.where(y == 1)

#Hacemos 3 arrays para separar los indices de las clases 0,1 y 2
x_2 = np.array([x[y0],x[y1]])

plt.scatter(x_2[0][:, 1], x_2[0][:, 2],  c = 'blue', label = '1')
plt.scatter(x_2[1][:, 1], x_2[1][:, 2],  c = 'orange', label = '5')
t = np.linspace(0,1, 100)
plt.plot( t, (-w_4[0]-w_4[1]*t)/w_4[2], c = 'red', label = "sgd tam 17")
plt.plot( t, (-w_pseudo[0]-w_pseudo[1]*t)/w_pseudo[2], c = 'green', label = "pseudoinverse")
plt.legend();
plt.title("Ejercicio1")

plt.figure()
plt.show()



input("\n--- Pulsar tecla para continuar ---\n")

w_1 = sgd(x,y,200,0.1, 100)
print ('Bondad del resultado para grad. descendente estocastico, tamaño de mini bach = 100:\n')
print ("Ein: ", Err(x,y,w_1))
print ("Eout: ", Err(x_test, y_test, w_1))
print("w: ", w_1)

w_2 = sgdN(x,y,200,0.1)
print ('Bondad del resultado para grad. descendente estocastico, tamaño de mini bach = 1:\n')
print ("Ein: ", Err(x,y,w_2))
print ("Eout: ", Err(x_test, y_test, w_2))
print("w: ", w_2)


y0 = np.where(y == -1)
y1 = np.where(y == 1)

#Hacemos 3 arrays para separar los indices de las clases 0,1 y 2
x_2 = np.array([x[y0],x[y1]])

plt.scatter(x_2[0][:, 1], x_2[0][:, 2],  c = 'blue', label = '1')
plt.scatter(x_2[1][:, 1], x_2[1][:, 2],  c = 'orange', label = '5')
t = np.linspace(0,1, 100)
plt.plot( t, (-w[0]-w[1]*t)/w[2], c = 'red', label = "sgd tam 32")
plt.plot( t, (-w_2[0]-w_2[1]*t)/w_2[2], c = 'purple', label = "sgd tam 1")
plt.plot( t, (-w_1[0]-w_1[1]*t)/w_1[2], c = 'black', label = "sgd tam 100")
plt.plot( t, (-w_4[0]-w_4[1]*t)/w_4[2], c = 'brown', label = "sgd tam 17")
plt.plot( t, (-w_pseudo[0]-w_pseudo[1]*t)/w_pseudo[2], c = 'green', label = "pseudoinverse")
plt.legend();
plt.title("Ejercicio1")

plt.figure()
plt.show()


input("\n--- Pulsar tecla para continuar ---\n")
#Ejercicio 2.2

print('Ejercicio 2\n')
# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))

def sign(x):
	if x >= 0:
		return 1
	return -1


#renombro porque hay una funcion f en el ejercicio 1
def g(x1, x2):
	return sign((x1-0.2)**2+x2**2-0.6) 

N = 1000 #tamaño de la muestra de puntos

entrenamiento = simula_unif(N, 2, 1) #genera una muestra de entrenamiento de tamaño N en dos dimensiones y en un cuadrado de lado 2

plt.scatter(entrenamiento[:,0], entrenamiento[:,1], c ='r') #pintamos dicha muestra
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")
 
etiquetas=[]

for i in range(0,N): # asignamos a cada elemnto su etiqueta mediante la funcion g
    etiquetas.append(g(entrenamiento[i,0], entrenamiento[i,1]))
    
etiquetas = np.asarray(etiquetas)

#generamos un array con posiciones aleatorias de tamaño el 10% de la muestra
posiciones = np.random.choice(etiquetas.size, int(0.1*etiquetas.size), replace=True)

for i in range(0, posiciones.size):
    etiquetas[posiciones[i]]=-etiquetas[posiciones[i]] #cambiamos el signo de dichas etiquetas
    
    
plt.scatter(entrenamiento[:,0], entrenamiento[:,1], c = etiquetas)
plt.show()  #pintamos la muestra diferenciando los elementos por sus etiquetas

input("\n--- Pulsar tecla para continuar ---\n")

vector_unos = np.ones((len(entrenamiento),1))
entrenamiento = np.concatenate((vector_unos, entrenamiento), axis = 1)
# generamos una matriz X según nos dice el enunciado, es decir, de la forma (1 x1 x2)

w = sgd(entrenamiento, etiquetas, 100, 0.01, 32)  #Hacemos gradiente estocastico con los datos generados
error_interno = Err(entrenamiento, etiquetas, w) #Calculamos el error interno

print('Error interno', error_interno)
print('w: ', w)

input("\n--- Pulsar tecla para continuar ---\n")

plt.scatter(entrenamiento[:,1], entrenamiento[:,2], c =etiquetas)
t = np.linspace(0,0.18, 100)
plt.plot( t, (-w[0]-w[1]*t)/w[2], c = 'red')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")
error_int = 0
error_ext = 0
for i in range(1000):
    #entrenamiento
    entrenamiento = simula_unif(N, 2, 1)
    etiquetas=[]    
    for i in range(0,N):
        etiquetas.append(g(entrenamiento[i,0], entrenamiento[i,1]))    
    etiquetas = np.asarray(etiquetas)
    posiciones = np.random.choice(etiquetas.size, int(0.1*etiquetas.size), replace=True)

    for i in range(0, posiciones.size):
        etiquetas[posiciones[i]]=-etiquetas[posiciones[i]]
    vector_unos = np.ones((len(entrenamiento),1))
    x_entren = np.concatenate((vector_unos, entrenamiento), axis = 1)

    w = sgd(x_entren, etiquetas, 100, 0.01, 32)
    error_int+= Err(x_entren, etiquetas, w) 
    
    
    #test
    test =  simula_unif(N, 2, 1)
    etiquetas=[]    
    for i in range(0,N):
        etiquetas.append(g(test[i,0], test[i,1]))    
    etiquetas = np.asarray(etiquetas)
    posiciones = np.random.choice(etiquetas.size, int(0.1*etiquetas.size), replace=True)
    for i in range(0, posiciones.size):
        etiquetas[posiciones[i]]=-etiquetas[posiciones[i]]
    
    vector_unos = np.ones((len(test),1))
    x_test = np.concatenate((vector_unos,test), axis = 1)
    
    error_ext+= Err(x_test, etiquetas, w)
    
error_medio_int = error_int / 1000
error_medio_ext = error_ext / 1000
print('Error medio interior', error_medio_int)
print('Error medio exterior', error_medio_ext)

input("\n--- Pulsar tecla para continuar ---\n")
#Repetimos el experimento con caracteristicas no lineales
print('Repetimos el experimento con caracteristicas no lineales')

entrenamiento = simula_unif(N, 2, 1) #obtenemos la muestra de entrenamiento
etiquetas=[]    
for i in range(0,N):
    etiquetas.append(g(entrenamiento[i,0], entrenamiento[i,1]))    #aplicamos g para obtener las etiquetas
etiquetas = np.asarray(etiquetas)
posiciones = np.random.choice(etiquetas.size, int(0.1*etiquetas.size), replace=True) #obtenemos las posiciciones para introducir un 10% de aleatoridad

for i in range(0, posiciones.size):
    etiquetas[posiciones[i]]=-etiquetas[posiciones[i]] #cambiamos las etiquetas del 10%
vector_unos = np.ones((len(entrenamiento),1))  #generamos la primera columna de la matriz x (columna de unos)
vector_mult = (entrenamiento[:,0]*entrenamiento[:,1]).reshape((-1,1)) #generamos la cuarta columna (x1*x2)
vector_x_cuadrado = (entrenamiento[:,0]**2).reshape((-1,1)) #generamos la quinta columna (x1^2)
vector_y_cuadrado = (entrenamiento[:,1]**2).reshape((-1,1)) #generamos la sexta columna (x2^2)
x_entren = np.concatenate((vector_unos, entrenamiento,vector_mult , vector_x_cuadrado, vector_y_cuadrado), axis = 1) #concatenamos las columnas construidas

w = sgd(x_entren, etiquetas, 100, 0.01, 32) #aplicamos gradiente descendiente estocastico
error_int = Err(x_entren, etiquetas, w)
print('Error interno: ', error_int)
print('w: ', w)


dominio_prov = np.linspace(-1,1, 100) #genero 100 números entre -1 y 1 equiespaciados
#me quedo con las posiciones en las que se cumple la desigualdad que nos dice que el valor de x pertenece al dominio
posiciones_dominio = np.where((w[2]**2+2*w[2]*w[3]*dominio_prov+w[3]**2*dominio_prov**2) >= (4*w[0]*w[5]+4*w[1]*w[5]*dominio_prov+4*w[4]*w[5]*dominio_prov**2))
t = dominio_prov[posiciones_dominio] #genero un vector con los valores de x que están en el dominio
        
plt.scatter(entrenamiento[:,0], entrenamiento[:,1], c =etiquetas) 
#pinto las dos gráficas, correspondientes a las dos soluciones de la ecuación
plt.plot( t,  (np.sqrt(-4*w[5]*t*w[1]+2*w[3]*t*w[2]+w[2]**2-4*w[0]*w[5]+w[3]**2*t**2-4*w[4]*w[5]*t**2)-w[2]-w[3]*t)/(2*w[5]), c = 'red')
plt.plot( t, -(np.sqrt(-4*w[5]*t*w[1]+2*w[3]*t*w[2]+w[2]**2-4*w[0]*w[5]+w[3]**2*t**2-4*w[4]*w[5]*t**2)-w[2]-w[3]*t)/(2*w[5]), c = 'red')

plt.show()    
    
input("\n--- Pulsar tecla para continuar ---\n")

error_int = 0
error_ext = 0
for i in range(1000):
    #entrenamiento
    entrenamiento = simula_unif(N, 2, 1)
    etiquetas=[]    
    for i in range(0,N):
        etiquetas.append(g(entrenamiento[i,0], entrenamiento[i,1]))    
    etiquetas = np.asarray(etiquetas)
    posiciones = np.random.choice(etiquetas.size, int(0.1*etiquetas.size), replace=True)

    for i in range(0, posiciones.size):
        etiquetas[posiciones[i]]=-etiquetas[posiciones[i]]
    vector_unos = np.ones((len(entrenamiento),1))
    vector_mult = (entrenamiento[:,0]*entrenamiento[:,1]).reshape((-1,1))
    vector_x_cuadrado = (entrenamiento[:,0]**2).reshape((-1,1))
    vector_y_cuadrado = (entrenamiento[:,1]**2).reshape((-1,1))
    x_entren = np.concatenate((vector_unos, entrenamiento,vector_mult , vector_x_cuadrado, vector_y_cuadrado), axis = 1)

    w = sgd(x_entren, etiquetas, 100, 0.01, 32)
    error_int+= Err(x_entren, etiquetas, w)
    
    
    #test
    test =  simula_unif(N, 2, 1)
    etiquetas=[]    
    for i in range(0,N):
        etiquetas.append(g(test[i,0], test[i,1]))    
    etiquetas = np.asarray(etiquetas)
    posiciones = np.random.choice(etiquetas.size, int(0.1*etiquetas.size), replace=True)
    
    for i in range(0, posiciones.size):
        etiquetas[posiciones[i]]=-etiquetas[posiciones[i]]
    
    vector_unos = np.ones((len(test),1))
    vector_mult = (test[:,0]*test[:,1]).reshape((-1,1))
    vector_x_cuadrado = (test[:,0]**2).reshape((-1,1))
    vector_y_cuadrado = (test[:,1]**2).reshape((-1,1))
    x_test = np.concatenate((vector_unos, test,vector_mult , vector_x_cuadrado, vector_y_cuadrado), axis = 1)

    
    
    error_ext+= Err(x_test, etiquetas, w)
    
error_medio_int = error_int / 1000
error_medio_ext = error_ext / 1000
print('Error medio interior', error_medio_int)
print('Error medio exterior', error_medio_ext)
