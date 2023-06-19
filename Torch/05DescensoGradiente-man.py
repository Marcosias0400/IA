#******************************************
"""
Instituto Politecnico Nacional
Escuela Superior de Fisica y Matematicas
Licenciatura de Matematica Algoritmica
Fundamentos de Inteligencia Artificial
Editor: Ortiz Ortiz Bosco
Titulo: Descenso de Gradiente Manual 
"""
import numpy as np 
#------------------------------------------
# Calcular cada paso manualmente
# Regresion Lineal
# f = w * x 
# Aqui : f = 2 * x
X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)
w = 0.0
#------------------------------------------
# Salida del modelo
def forward(x):
    return w * x
# costo = MSE
def loss(y, y_pred):
    return ((y_pred - y)**2).mean()
# J = MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N * 2x(w*x - y)
def gradient(x, y, y_pred):
    return np.mean(2*x*(y_pred - y))
print(f'Prediccion antes del entrenamiento: f(5) = {forward(5):.3f}')
# entrenamiento
learning_rate = 0.01
n_iters = 20
for epoch in range(n_iters):
    # prediccion = paso de evaluacion
    y_pred = forward(X)
    # Costo
    l = loss(Y, y_pred)
    # Calculo de gradientes
    dw = gradient(X, Y, y_pred)
    # Actualizacion de pesos
    w -= learning_rate * dw
    if epoch % 2 == 0:
        print(f'iteracion {epoch+1}: w = {w:.3f}, costo = {l:.8f}')
     
print(f'Prediccion despues del entrenamiento: f(5) = {forward(5):.3f}')
#******************************************
