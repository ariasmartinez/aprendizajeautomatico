# -*- coding: utf-8 -*-
"""
Exercise 2 
Author: Blanca Cano Camaro
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

print('_______LINEAR REGRESSION EXERCISE _______\n')
print('Exercise 1\n')
#input('\n Enter to start\n') UNCOMMENT


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
def Error(x,y,w):
    '''quadratic error 
    INPUT
    x: input data matrix
    y: target vector
    w:  vector to 

    OUTPUT
    quadratic error >= 0
    '''
    error_times_n = np.linalg.norm(x.dot(w) - y.reshape(-1,1))
  
    return error_times_n/len(y)


def dError(x,y,w):
    ''' partial derivative
    '''
    
    return (2/len(x)*(x.T.dot(x.dot(w) - y.reshape(-1,1))))

# Gradiente Descendente Estocastico
def sgd(x,y, eta = 0.01, max_iter = 1000, batch_size = 32):
    '''
    Stochastic gradeint descent
    x: data set
    y: target vector
    eta: learning rate
    max_iter     
    '''

    w = np.zeros((x.shape[1], 1), np.float64)
    #print( f'EN SGD LA W VALE {w}')
    n_iterations = 0

    len_x = len(x)
    x_index = np.arange( len_x )
    batch_start = 0

    while n_iterations < max_iter:
            
            #shuffle and split the same into a sequence of mini-batches
        if batch_start == 0:
                x_index = np.random.permutation(x_index)
        iter_index = x_index[ batch_start : batch_start + batch_size]

        w = w - eta* dError(x[iter_index, :], y[iter_index], w)
       
        n_iterations += 1

        batch_start += batch_size
        if batch_start > len_x:  # Si hemos llegado al final reinicia
                batch_start = 0

    return w

def pseudoInverseMatrix ( X ):
    '''
    input: 
    X: is a matrix (must be a np.array) to use transpose and dot method
    return: hat matrix 
    '''

    '''
    #S =( X^TX ) ^{-1}
    simetric_inverse = np.linalg.inv( X.T.dot(X) )

    # S X^T = ( X^TX ) ^{-1} X^T
    return simetric_inverse.dot(X.T)
    '''
    return np.linalg.pinv(X)


# Pseudoinverse	
def pseudoInverse(X, Y):
    ''' TO-DO matrix dimension is correct?
    input:
    X is  matrix, R^{m} \time R^{m} 
    Y is a vector (y_1, ..., y_m)
    
    OUTPUT: 
    w: weight vector
    '''
    X_pseudo_inverse = pseudoInverseMatrix ( X )
    Y_transposed = Y.reshape(-1, 1)
    
    w = X_pseudo_inverse.dot( Y_transposed)
    
    return w


# Evaluating the autput

def performanceMeasurement(x,y,w):
    '''Evaluating the output binary case

    OUTPUT: 
    bad_negative, bad_positives, input_size
    '''

    sign_column = np.sign(x.dot(w)) - y.reshape(-1,1)

    bad_positives = 0
    bad_negatives = 0
    
    for sign in sign_column[:,0]:
        if sign > 0 :
                bad_positives += 1
        elif sign < 0 :
                bad_negatives += 1

    input_size = len(y)

    return bad_negatives, bad_positives, input_size

def evaluationMetrics (x,y,w, label = None):
    '''PRINT THE PERFORMANCE MEASUREMENT
    '''
    bad_negatives, bad_positives, input_size = performanceMeasurement(x,y,w)

    accuracy = ( input_size-(bad_negatives +bad_positives))*100 / input_size

    if label :
        print(label)
    print ( 'Input size: ', input_size )    
    print( 'Bad negatives :', bad_negatives)
    print( 'Bad positives :', bad_positives)
    print( 'Accuracy rate :', accuracy, '%')





    
## Draw de result

### scatter plot
def plotResults (x,y,w, title = None):
        label_5 = 1
        label_1 = -1

        labels = (label_5, label_1)
        colors = {label_5: 'b', label_1: 'r'}
        values = {label_5: 'Number 5', label_1: 'Number 1'}

        plt.clf()

        # data set plot 
        for number_label in labels:
                index = np.where(y == number_label)
                plt.scatter(x[index, 1], x[index, 2], c=colors[number_label], label=values[number_label])

        # regression line
        symmetry_for_cero_intensity = -w[0]/w[2]

        # en el caso de x1 = 1, tenemos 0 = w0 + w1 * w2 * x2
        # luego x2 = (-w0 - w1) /w2
        symmetry_for_one_intensity= (-w[0] - w[1])/w[2]
        plt.plot([0, 1], [symmetry_for_cero_intensity, symmetry_for_one_intensity], 'k-', label=(title+ ' regression'))

                

        if title :
                plt.title(title)
        plt.xlabel('Average intensity')
        plt.ylabel('Simmetry')
        plt.legend()
        plt.show()

### Draw a line ( it is a regression os a line so must have one line

### _____________ DATA ____________________

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

print("\n___ Goodness of the Stochastic Gradient Descendt (SGD) fit ___\n")

batch_sizes = [2,32, 200, 15000]
for _batch_size in batch_sizes:
        w = sgd(x,y, eta = 0.1, max_iter = 200, batch_size = _batch_size)

        _title = f'SGD, batch size {_batch_size}'
        print( '\n\t'+_title)
        print ("Ein: ", Error(x,y,w))
        print ("Eout: ", Error(x_test, y_test, w))
        evaluationMetrics (x,y,w, '\nEvaluating output training data set')
        evaluationMetrics (x_test, y_test, w, '\nEvaluating output test data set')
        plotResults(x,y,w, title = _title)
        if (_batch_size == 32):
            print('w sgd: ', w)


w_pseudoinverse = pseudoInverse(x, y) 
print("\n___ Goodness of the Pseudo-inverse fit ___\n")
print("  Ein:  ", Error(x, y, w_pseudoinverse))
print("  Eout: ", Error(x_test, y_test, w_pseudoinverse))
print('w pseudo: ', w_pseudoinverse)

evaluationMetrics (x,y,w, '\nEvaluating output training data set')
evaluationMetrics (x_test, y_test, w, '\nEvaluating output test data set')
plotResults(x,y,w, title = 'Pseudo-inverse')


input("\n--- Type any key to continue ---\n")

#Seguir haciendo el ejercicio...
'''
print('Ejercicio 2\n')
# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))

def sign(x):
	if x >= 0:
		return 1
	return -1

def f(x1, x2):
	return sign(?) 
'''
#Seguir haciendo el ejercicio...

