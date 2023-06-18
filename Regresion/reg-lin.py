#******************************************
"""
Instituto Politecnico Nacional
Escuela Superior de Fisica y Matematicas
Licenciatura de Matematica Algoritmica
Fundamentos de Inteligencia Artificial
Editor: Ortiz Ortiz Bosco
Titulo: Regresion Lineal (a mano)
"""
#------------------------------------------
# Modulos Importados
import matplotlib.pyplot as plt
import numpy as np
#------------------------------------------
# Genera datos de Prueba
mm=3
bb=5
#------------------------------------------
x=np.linspace(0,1,400,dtype=np.float32)
e=np.random.normal(0,0.1,400)
y=mm*x+bb+e
#------------------------------------------
x1=np.linspace(1,2,20,dtype=np.float32)
e1=np.random.normal(0,0.1,20)
y1=mm*x1+bb+e1
#------------------------------------------
m=0
b=0
#------------------------------------------
# tasa de aprendizaje
L=0.1
#------------------------------------------
# Iteraciones 
iter=200
#------------------------------------------
# Minimizar recorriendo el -gradiente
n=float(len(x))
for i in range(iter):
  # Prediccion en Y
  Y_pred=m*x+b 
  # Parcial con respecto a m
  D_m=(-2/n)*sum(x*(y-Y_pred)) 
  # Parcial con respecto a b
  D_b=(-2/n)*sum(y-Y_pred)
  # Mejorar m
  m-=L*D_m
  # Mejorar b
  b-=L*D_b
print(m,b)
yy=m*x1+b
#------------------------------------------
# Graficas
plt.scatter(x1,y1,color='green')
plt.plot(x1,yy,color='red',linewidth=1)
plt.show()
#******************************************
