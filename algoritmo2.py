import numpy as np
import time
import matplotlib.pyplot as plt
import pycutest
from numba import jit
import sys

@jit(nopython=True)
def newton_cauchy(xk,delta_k,gk,Bk):
	'''
	Funcion para calcular el paso de Cauchy.

	Toma como argumentos
	Punto xk (vector)
	radio de la region delta_k (escalar)
	gradiente gk (vector)
	Aproximacion de Hessiano Bk
	'''

	pk=np.zeros(len(xk))

	#Paso de Newton
	pk_N=np.linalg.solve(Bk,-gk)
	norm_gk=np.linalg.norm(gk)

	if np.linalg.norm(pk_N)<= delta_k:
	    # Si estamos dentro de la región consideramos el paso de Newton
	    pk=pk_N
	else: 
	    # Si estamos en la frontera o fuera, tomamos el paso de Cauchy
	    aux=gk.T @ Bk @ gk

	    if aux <= 0:
	        tau=1
	    else:
	        tau=min(1, norm_gk**3/(delta_k*aux) )

	    #Paso de Cauchy
	    pk=-(tau * (delta_k/norm_gk))*gk
	return pk

@jit(nopython=True)
def m(xk,sk,fk, gk, Bk):
	'''
	Funcion que regresa el valor del modelo cuadratico evaluado en un punto

	Toma como argumentos
	p: el punto donde se evalua (vector)
	f: la funcion que aproxima (escalar)
	g: el gradiente de f (vector)
	H: el Hessiano de f (matriz)
	'''
	return fk+ np.dot(gk.T,sk)+(1/2)*sk.T@Bk@sk

#@jit(nopython=True)
def mat_pos_def(H,beta = 0.001):
	'''
	Modificacion del Hessiano para que sea positivo definido

	Toma como argumento H el Hessiano (matriz)
	'''
	MIN = min(np.diag(H))
	if MIN > 0:
		T = 0
	else:
		T = -MIN + beta

	j = 0
	while True:

		try:
			#print('Cholesky',j)
			j += 1
			L = np.linalg.cholesky(H + T*np.identity(len(H)))
			#print('Exito Cholesky')
			return H + T*np.identity(len(H))
		except np.linalg.LinAlgError: # Equivalente a "si cholesky falla"
			T = max(2*T,beta)

def get_rho(p,m,x,y,sk):
    '''
    Funcion que calcula la medida de ajuste del modelo

    Argumentos:
    f: funcion a evaluar
    grad_f: funcion que contiene el gradiente de f
    hess_f: funcion que contiene el Hessiano de f
    m: funcion que contiene el modelo que aproxima a f
    x: punto de evaluacion (vector)
    y: segundo punto de evaluacion (x + sk)
    sk: direccion del nuevo punto (vector)
    '''
    fx,gx= p.obj(x, gradient=True)
    fy,gy= p.obj(y, gradient=True)
    Bx,By=p.hess(x),p.hess(y)

    num=fx-fy
    denom=m(x,np.zeros(len(x)),fx, gx, Bx)-m(x,sk,fy, gy, By)
    return num/denom

def Cauchy(x,g,B,delta):
	'''
	Funcion para calcular el paso de Cauchy.

	Toma como argumentos
	Punto x (vector)
	gradiente g (vector)
	Aproximacion de Hessiano B
	radio delta de la region (escalar)
	
	'''
	norma_g = np.linalg.norm(g)
	H_inv = np.linalg.inv(B)

	# Minimizador del modelo cuadratico sin restricciones
	pNewt = -np.dot(H_inv,g)
	norma_p = np.linalg.norm(pNewt)

	if norma_p <= delta: # Si el pNewt esta en la region de confianza
		return pNewt
	else: # Si no, usamos el paso de Cauchy
		pS = -delta/norma_g*g # Direccion de maximo descenso en la RC
		Producto = np.dot(np.transpose(g),np.dot(B,g))
		if Producto <= 0:
			return pS
		else:
			return min(1,norma_g**3/(delta*Producto))*pS

def Dogleg(x,g,B,delta):
	'''
	Funcion para calcular el paso de Dogleg.

	Toma como argumentos
	punto x (vector)
	gradiente g (vector)
	Aproximacion del Hessiano B (matriz)
	radio de la region delta (escalar)
	'''
	norma_g = np.linalg.norm(g)
	H_inv = np.linalg.inv(B)
	Producto = np.dot(np.transpose(g),np.dot(B,g))

	# Direccion de maximo descenso
	pU = -norma_g**2/Producto*g
	norma_pU = np.linalg.norm(pU)

	# Minimizador del modelo cuadratico sin restricciones
	pB = -np.dot(H_inv,g)
	norma_pB = np.linalg.norm(pB)


	if norma_pB <= delta: # Si el mejor paso esta dentro de la RC
		return pB

	elif norma_pU >= delta: # Si la pU esta fuera de la RC
		return Cauchy(x,g,B,delta)
	
	else: # Si pU esta dentro de RC y pB fuera
		a = np.linalg.norm(pB - pU)**2
		b = 2*np.dot(np.transpose(pB),pB-pU)
		c = norma_pU**2-delta**2

		tau = (-b+np.sqrt(b**2-4*a*c))/(2*a) + 1

		if tau >= 0 and tau <= 1:
			return tau*pU
		elif tau > 1 and tau <= 2:
			return pU + (tau - 1)*(pB - pU)
		else:
			print('Tau no esta entre 0 y 2')
#######_____________________Algoritmo 2_________________________________

def Forwtracking(x,s,p,modelo,rho, NIter = 100):
    '''
    Funcion que maximiza la medida del ajuste por medio de forward tracking

    Argumentos:
    Punto de evaluacion x (vector)
    Direccion s (vector)
    Funcion a evaluar func
    Funcion que contiene el gradiente de func, grad
    Funcion que contiene el Hessiano de func, Hes
    Funcion que contiene el modelo que aproxima a func, modelo
    Constante de aumento de alfa, rho
    '''
    #Contadores de evaluaciones
    contf=0
    contg=0
    
    
    a = 1
    alfa = rho*a
    k = 0

    # Evaluaciones de funciones
    f,g=p.obj(x, gradient=True)
    contf, contg = contf+1, contg+1
    B=p.hess(x)
    mod = m(x,np.zeros((len(x),1)),f, g, B)

    fa,ga=p.obj(x+a*s, gradient=True)
    contf, contg = contf+1, contg+1
    Ha=p.hess(x+a*s)
    Ha = mat_pos_def(Ha)

    falfa,galfa=p.obj(x+alfa*s, gradient=True)
    contf, contg = contf+1, contg+1
    Halfa=p.hess(x+alfa*s)
    Halfa = mat_pos_def(Halfa)

    # Calculo de las medidas de ajuste
    m1 = mod - m(x,a*s,fa,ga,Ha)
    m2 = mod - m(x,alfa*s,falfa,galfa,Halfa)
    phi1 = (f - fa)/m1
    phi2 = (f - falfa)/m2

    # Ciclo principal con los pasos anteriores
    while phi2 > phi1 and m2 < m1 and k < NIter:
        a = alfa
        alfa = rho*alfa

        fa,ga=p.obj(x+a*s, gradient=True)
        Ha=p.hess(x+a*s)
        Ha = mat_pos_def(Ha)

        falfa,galfa=p.obj(x+alfa*s, gradient=True)
        Halfa=p.hess(x+alfa*s)
        Halfa = mat_pos_def(Halfa)
        
        contf, contg = contf+2, contg+2

        m1 = mod - m(x,a*s,fa,ga,Ha)
        m2 = mod - m(x,alfa*s,falfa,galfa,Halfa)
        phi1 = (f - fa)/m1
        phi2 = (f - falfa)/m2

        k += 1
    return a, contf, contg

def RegConf2(x0,modelo,p,d0,eta1 = 0.1,eta2 = 0.9, gamma1 = 0.5, gamma2 = 1, gamma3 = 3, tol = 1e-5, NIter = 100000):
	'''
	Algoritmo 2 del articulo

	Argumentos:
	Punto inicial x (vector)
	Funcion con el modelo que aproxima, modelo
	Funcion objetivo, func
	Funcion con el gradiente de func, grad
	Funcion con el Hessiano de func, Hes
	Radio inicial, d0
	Umbral de aceptacion eta1 y eta2
	Factores que modifican el radio: gamma1, gamma2, gamma3
	Tolerancia, tol
	Numero maximo de iteraciones NIter
	'''
	# Contadores de evaluaciones
	contf=0
	contg=0

	# Evaluaciones de funciones
	x = np.copy(x0)
	f,g = p.obj(x,gradient = True)
	contf, contg = contf+1, contg+1
	H = p.hess(x)
	B = mat_pos_def(H)
	norma_g = np.linalg.norm(g)

	# Ciclo principal
	k = 0
	while norma_g > tol and k < NIter:

		s = Dogleg(x,g,B,d0) # Determina la direccion del nuevo paso
		#s = Steihaug(g,H,B,d0)
		norma_s = np.linalg.norm(s)

		rho = get_rho(p,m,x,x+s,s) # Calcula la medida del ajuste
		contf, contg = contf+2, contg+2

		if rho >= eta2 and norma_s == d0: 
			# acepta nuevo paso y aumenta delta por forwardtracking
			a = Forwtracking(x,s,p,modelo,1+1e-4)
			contf, contg = contf+a[1], contg+a[2]
			# Algoritmo de Forwardtracking
			a=a[0]
			x0 = np.copy(x)
			x = x + a*s
			d0 = np.linalg.norm(x0 - x)

		if rho >= eta2 and norma_s < d0:
			#acepta nuevo paso y aumenta delta por gamma3
			d0 = gamma3*d0
			x = x + s

		elif rho >= eta1:
			# Acepta nuevo paso y reduce ligeramente (o conserva) a delta
			d0 = gamma2*d0
			x = x + s

		else:
			# rechaza el nuevo paso, reduce delta
			d0 = gamma1*d0

		k += 1

		f,g = p.obj(x,gradient = True)
		contf, contg = contf+1, contg+1
		H = p.hess(x)
		B = mat_pos_def(H)
		norma_g = np.linalg.norm(g)
		print(k,norma_g)

	return x,k,contf, contg


##______________________________________________________________________
a_file = open("problemas2.txt", "r")
lista = []
for line in a_file:
	
	stripped_line = line.strip()
	lista.append(stripped_line)
#lista=['CRAGGLVY']	
for n in lista:
    p = pycutest.import_problem(n)
    x = p.x0
    _,gx= p.obj(x, gradient=True)
    delta = 0.1*np.linalg.norm(gx)
    #_________________________
    start = time.process_time()
    x_op=RegConf2(x,m,p,delta)
    end = time.process_time()
    #___________________
    fx,gx= p.obj(x_op[0], gradient=True)
    print(n,',',len(x),',',end-start,',',x_op[1],',',x_op[2],',',x_op[3],',', x_op[2]+len(x)*x_op[3],',',fx,',',np.linalg.norm(gx) )

    

 
