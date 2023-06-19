#******************************************
"""
Instituto Politecnico Nacional
Escuela Superior de Fisica y Matematicas
Licenciatura de Matematica Algoritmica
Fundamentos de Inteligencia Artificial
Editor: Ortiz Ortiz Bosco
Titulo: Descenso de gradiente automatico
"""
import torch
#------------------------------------------
# reemplazaremos el calculo manual de 
# gradiente con el parametro autograd 
# Regresion Lineal
# f = w * x 
# Aqui : f = 2 * x
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
# Salida del Modelo
def forward(x):
    return w * x
# costo = MSE
def loss(y, y_pred):
    return ((y_pred - y)**2).mean()
print(f'Predicion antes del entrenamiento: f(5) = {forward(5).item():.3f}')
#------------------------------------------
# Entrenamiento
learning_rate = 0.01
n_iters = 100
for epoch in range(n_iters):
    # prediccion = evaluacion
    y_pred = forward(X)
    # Perdida
    l = loss(Y, y_pred)
    # calculo de gradiente = paso backward 
    l.backward()
    # actualizacion de pesos
    # w.datos = w.datos 
    #- learning_rate * w.grad
    with torch.no_grad():
        w -= learning_rate * w.grad
    # los gradientes e cero despues de 
    # actualizar
    w.grad.zero_()
    if epoch % 10 == 0:
        print(f'iteracion {epoch+1}: w = {w.item():.3f}, costo = {l.item():.8f}')

print(f'Prediccion despues del entrenamiento: f(5) = {forward(5).item():.3f}')
#******************************************
