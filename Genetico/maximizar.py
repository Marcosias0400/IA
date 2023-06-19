#******************************************
"""
Instituto Politecnico Nacional
Escuela superior de fisica y Matematicas
licenciatura de Matematica algoritmica
Fundamentos de inteligencia Artificial
Editor: Ortiz Ortiz Bosco
titulo: Algoritmo Genetico; Maximo de una 
funcion
"""
#------------------------------------------
# Modulos a importar
import matplotlib.pyplot as plt
import numpy as np
import math as mt
#------------------------------------------
# funcion de muchos maximos
def f_x(x):
  return -(0.1+(1-x)**2-0.1*mt.cos(6*mt.pi*(1-x)))+2
#------------------------------------------
# Lista decimal
def listToDecimal(num):
  decimal:np.float64=0
  for i in range(len(num)):
    decimal+=num[i]*10**(-i)
  return decimal
#------------------------------------------
# Mutaciones
def mutar(individuos,prob,pool):
  for j in range(len(individuos)):
    mutar_individuo=individuos[j]
    if np.random.random()<prob:
      mutacion=np.random.choice(pool[0])
      mutar_individuo=[mutacion]+mutar_individuo[1:]
    for k in range(1,len(mutar_individuo)):
      if np.random.random()<prob:
        mutacion=np.random.choice(pool[1])
        mutar_individuo=mutar_individuo[0:k]+[mutacion]+mutar_individuo[k+1:]
    individuos[j]=mutar_individuo
#------------------------------------------
# PROGRAMA PRINCIPAL
eje_x=np.arange(0,2,0.02)
eje_y=np.array(list(map(f_x,eje_x)))
#------------------------------------------
# Nucleotidos
tam_ind=15
tarja_genetica=[[0,1],[0,1,2,3,4,5,6,7,8,9]]
#------------------------------------------
# Poblacion
poblacion=[]
for l in range(100):
  individuo=[]
  individuo+=[np.random.choice(tarja_genetica[0])]
  individuo+=list(np.random.choice(tarja_genetica[1],tam_ind-1))
  np.array(individuo)
  poblacion.append(individuo)
np.array(poblacion)
#------------------------------------------
# Evolucion
tam_poblacion=len(poblacion)
generaciones=300
for _ in range(generaciones):
  adelgazamiento=[]
  for individuo in poblacion:
    x=listToDecimal(individuo)
    y=f_x(x)
    adelgazamiento+=[y]
  adelgazamiento=np.array(adelgazamiento)
  adelgazamiento=adelgazamiento/adelgazamiento.sum()
  offspring=[]
  for m in range(tam_poblacion//2):
    parientes=np.random.choice(tam_poblacion,2,p=adelgazamiento)
    punto_cruza=np.random.randint(tam_ind)
    offspring+=[poblacion[parientes[0]][:punto_cruza]+poblacion[parientes[1]][punto_cruza:]]
    offspring+=[poblacion[parientes[1]][:punto_cruza]+poblacion[parientes[0]][punto_cruza:]]
  np.array(offspring)
  poblacion=offspring
  mutar(poblacion,0.005,tarja_genetica)
#------------------------------------------
# El mas apto
n=np.where(adelgazamiento==adelgazamiento.max())
if np.size(n[0])==1:
  print(listToDecimal(poblacion[int(n[0])]))
else: 
  print("La solucion no fue unica")
#------------------------------------------
# Grafica final
for individuo in poblacion:
  x=listToDecimal(individuo)
  y=f_x(x)
  plt.plot(x,y,'x')
plt.plot(eje_x,eje_y)
plt.show()
#******************************************
