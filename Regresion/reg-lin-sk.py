#******************************************
"""
Instituto Politecnico Nacional
Escuela Superior de Fisica y Matematicas
Licenciatura de Matematica Algoritmica
Fundamentos de Inteligencia Artificial
Editor: Ortiz Ortiz Bosco
Titulo: Regresion Lineal usando sklearn
"""
#------------------------------------------
# Modulos Importados
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,r2_score
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
x=np.reshape(x,(400,1))
x1=np.reshape(x1,(20,1))
#------------------------------------------
# Crear Objeto de Regresion lineal
regr_lin=linear_model.LinearRegression()
#------------------------------------------
# Entrenar Modelo
regr_lin.fit(x,y)
#------------------------------------------
# Prediccion
yy=regr_lin.predict(x1)
#------------------------------------------
# Los coeficientes
print("Coeficientes:\t",regr_lin.coef_)
#------------------------------------------
# Error medio cuadrado
print("error medio caudrado: %.2f" %mean_squared_error(y1,yy))
#------------------------------------------
# Coeficientes de determinacion: 1 es
# Prediccion perfecta=1
print("Coeficiente de determinacion: %.2f" %r2_score(y1,yy))
#------------------------------------------
# Graficas
plt.scatter(x1,y1,color='green')
plt.plot(x1,yy,color='red',linewidth=1)
plt.show()
#******************************************
