#******************************************
"""
Instituto Politecnico Nacional
Escuela Superior de Fisica y Matematicas
Licenciatura de Matematica Algoritmica
Fundamentos de Inteligencia Artificial
Editor: Ortiz Ortiz Bosco 
Titulo: Costo y optimizador
"""
#------------------------------------------
# 1)Disenyo de modelo (entrada,salida, 
# evaluacion en diferentes capas)
# 2)Construir funcion de costo y optimizador
# 3)Ciclo de entrenamiento
#    -Evaluacion=Calcula la evaluacion y costo
#    -Derivada=Calculo de gradiente
#    -Actualizacion de pesos
import torch
import torch.nn as nn
# REgresion Lineal
# f = w * x 
# Aqui: f = 2 * x
#------------------------------------------
# 0)Muestras de entrenamiento
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
# 1)disenyo de modelo:funcion de optimizacion
# y optimizacion de pesos
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
def forward(x):
    return w * x
print(f'Prediccion antes del entrenamiento: f(5) = {forward(5).item():.3f}')
# 2) Define costo y optimizador
learning_rate = 0.01
n_iters = 100
# funcion llamable
loss = nn.MSELoss()
optimizer = torch.optim.SGD([w], lr=learning_rate)
# 3)Ciclo de entrenamiento
for epoch in range(n_iters):
    # prediccion=paso de evaluacion
    y_predicted = forward(X)
    # costo
    l = loss(Y, y_predicted)
    # calclo de gradientes=paso de derivacion
    l.backward()
    # actualiza pesos
    optimizer.step()
    # despues de actualizacion, vaciar 
    # el gradiente
    optimizer.zero_grad()
    if epoch % 10 == 0:
        print('iteracion', epoch+1, ': w = ', w, ' costo = ', l)

print(f'Prediccion despues del entrenamiento: f(5) = {forward(5).item():.3f}')
#******************************************
