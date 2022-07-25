import numpy as np
import matplotlib.pyplot as plt

Alg1 = np.loadtxt('salidas.txt',delimiter = ',',usecols = (1,2,3,4,5,6,7,8))
Alg2 = np.loadtxt('salidas2.txt',delimiter = ',',usecols = (1,2,3,4,5,6,7,8))
Alg3 = np.loadtxt('salidas3.txt',delimiter = ',',usecols = (1,2,3,4,5,6,7,8))

rT = np.zeros((len(Alg1[:,1]),3))
rF = np.zeros((len(Alg1[:,1]),3))
rG = np.zeros((len(Alg1[:,1]),3))
rFG = np.zeros((len(Alg1[:,1]),3))

for i in range(len(Alg1[:,1])):
	minimo = min(Alg1[i,1],Alg2[i,1],Alg3[i,1])
	rT[i,0] = Alg1[i,1]/minimo
	rT[i,1] = Alg2[i,1]/minimo
	rT[i,2] = Alg3[i,1]/minimo

	minimo = min(Alg1[i,3],Alg2[i,3],Alg3[i,3])
	rF[i,0] = Alg1[i,3]/minimo
	rF[i,1] = Alg2[i,3]/minimo
	rF[i,2] = Alg3[i,3]/minimo

	minimo = min(Alg1[i,4],Alg2[i,4],Alg3[i,4])
	rG[i,0] = Alg1[i,4]/minimo
	rG[i,1] = Alg2[i,4]/minimo
	rG[i,2] = Alg3[i,4]/minimo

	minimo = min(Alg1[i,5],Alg2[i,5],Alg3[i,5])
	rFG[i,0] = Alg1[i,5]/minimo
	rFG[i,1] = Alg2[i,5]/minimo
	rFG[i,2] = Alg3[i,5]/minimo

def rho(sigma,datos):
	n = len(datos)
	suma = 0
	for i in range(n):
		if datos[i] <= sigma:
			suma += 1
		
	return suma/n

# Evaluaciones totales

sigmaFG = np.arange(1,1200,0.01)
rhoFG1 = np.zeros(len(sigmaFG))
rhoFG2 = np.zeros(len(sigmaFG))
rhoFG3 = np.zeros(len(sigmaFG))
for i in range(len(sigmaFG)):
	rhoFG1[i] = rho(sigmaFG[i],rFG[:,0])
	rhoFG2[i] = rho(sigmaFG[i],rFG[:,1])
	rhoFG3[i] = rho(sigmaFG[i],rFG[:,2])

plt.plot(sigmaFG,rhoFG1,sigmaFG,rhoFG2,sigmaFG,rhoFG3)
plt.legend(('Algoritmo 1','Algoritmo 2','Algoritmo tradicional'))
plt.xlabel('sigma')
plt.ylabel('rho')
plt.show()

# Evaluaciones de F

sigmaF = np.arange(1,1200,0.01)
rhoF1 = np.zeros(len(sigmaF))
rhoF2 = np.zeros(len(sigmaF))
rhoF3 = np.zeros(len(sigmaF))
for i in range(len(sigmaF)):
	rhoF1[i] = rho(sigmaF[i],rF[:,0])
	rhoF2[i] = rho(sigmaF[i],rF[:,1])
	rhoF3[i] = rho(sigmaF[i],rF[:,2])

plt.plot(sigmaF,rhoF1,sigmaF,rhoF2,sigmaF,rhoF3)
plt.legend(('Algoritmo 1','Algoritmo 2','Algoritmo tradicional'))
plt.xlabel('sigma')
plt.ylabel('rho')
plt.show()

# Evaluaciones de G

sigmaG = np.arange(1,1200,0.01)
rhoG1 = np.zeros(len(sigmaG))
rhoG2 = np.zeros(len(sigmaG))
rhoG3 = np.zeros(len(sigmaG))
for i in range(len(sigmaG)):
	rhoG1[i] = rho(sigmaG[i],rG[:,0])
	rhoG2[i] = rho(sigmaG[i],rG[:,1])
	rhoG3[i] = rho(sigmaG[i],rG[:,2])

plt.plot(sigmaG,rhoG1,sigmaG,rhoG2,sigmaG,rhoG3)
plt.legend(('Algoritmo 1','Algoritmo 2','Algoritmo tradicional'))
plt.xlabel('sigma')
plt.ylabel('rho')
plt.show()

# Tiempo

sigmaT = np.arange(1,1200,0.01)
rhoT1 = np.zeros(len(sigmaT))
rhoT2 = np.zeros(len(sigmaT))
rhoT3 = np.zeros(len(sigmaT))
for i in range(len(sigmaT)):
	rhoT1[i] = rho(sigmaT[i],rT[:,0])
	rhoT2[i] = rho(sigmaT[i],rT[:,1])
	rhoT3[i] = rho(sigmaT[i],rT[:,2])

plt.plot(sigmaT,rhoT1,sigmaT,rhoT2,sigmaT,rhoT3)
plt.legend(('Algoritmo 1','Algoritmo 2','Algoritmo tradicional'))
plt.xlabel('sigma')
plt.ylabel('rho')
plt.show()