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

def TT_2(x0,p,m,Delta,eta_1=0.1,eta_2=0.9, gamma_1=0.5, gamma_2=1, gamma_3=3, eps=1e-5,itermax=10000):
    
    #Contadores de evaluaciones
    contf=0
    contg=0
    dimx=len(x0)
    
    
    k=0
    j=0

    x=x0

    _,g = p.obj(x, gradient=True)
    contf, contg =contf+1, contg+1
    g_norm=np.linalg.norm(g) 

    while g_norm > eps and k<= itermax:
        x_old=x
        f, aux = p.obj(x, gradient=True)
        contf, contg =contf+1, contg+1
        norm=np.linalg.norm(aux)


        # Calcula s_k
        _,gx = p.obj(x, gradient=True)
        contf, contg =contf+1, contg+1
        Hx = p.hess(x)
        Hx = mat_pos_def(Hx)
        s_k=Dogleg(x,gx,Hx,Delta)
        #
        #newton_cauchy(x,Delta,gx,Hx)

        # Calcula la medida de ajuste
        rho=get_rho(p,m,x_old,x+s_k,s_k)
        contf, contg =contf+2, contg+2

        # Calcula la norma de s_k
        sk_norm=np.linalg.norm(s_k)

        #--------------- Que tan bueno es x_k+s_k? ---------------

        # Punto en la frontera de la RC
        if rho >= eta_2 and sk_norm==Delta:
            #print('Muy bueno')
            Delta=Delta*(1+gamma_1/3)**(j+1)
            j=j+1
            k=k+1
            # Muy buena iteracion
            # Hay que calcular s_k de nuevo
            continue

        elif rho>=eta_2 and sk_norm<=Delta:
            # aumenta Delta y se acepta nuevo paso
            Delta=gamma_3*Delta
            #print('Bueno')
            x=x+s_k
            _,g = p.obj(x, gradient=True)
            contf, contg =contf+1, contg+1
            g_norm=np.linalg.norm(g)
            k=k+1
            j=j+1
            print(k,':',g_norm)
            continue

        elif eta_1 <= rho and rho < eta_2:
            # Acepta el nuevo paso y reduce ligeramente (o conserva) Delta
            #print('Regular')
            Delta=Delta*gamma_2
            x=x+s_k
            _,g = p.obj(x, gradient=True)
            contf, contg =contf+1, contg+1
            g_norm=np.linalg.norm(g)
            k=k+1
            j=j+1
            print(k,':',g_norm)
            continue

        elif rho< eta_1 and j>=1:
            # Acepta nuevo paso y mantiene Delta
            #print('No tan mal')
            x=x+s_k
            _,g = p.obj(x, gradient=True)
            contf, contg =contf+1, contg+1
            g_norm=np.linalg.norm(g)
            k=k+1
            j=j+1
            print(k,':',g_norm)
            continue

        elif rho< eta_1 and j==0:
            # Rechaza nuevo paso y reduce Delta
            #print('Mal')
            Delta=Delta*gamma_1
            _,g = p.obj(x, gradient=True)
            contf, contg =contf+1, contg+1
            g_norm=np.linalg.norm(g)
            k=k+1
            j=j+1
            print(k,':',g_norm)
            continue
    return x,k,contf,contg,dimx

a_file = open("problemas.txt", "r")
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
    #____llamada de la función___
    start = time.process_time()
    x_op=TT_2(x,p,m,delta)
    end = time.process_time()
    #___________________________
    fx,gx= p.obj(x_op[0], gradient=True)
    print(n,',',len(x),',',end-start,',',x_op[1],',',x_op[2],',',x_op[3],',', x_op[2]+len(x)*x_op[3],',',fx,',',np.linalg.norm(gx) )
  

    

