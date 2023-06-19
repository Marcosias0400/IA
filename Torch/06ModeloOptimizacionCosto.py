#******************************************
"""
Instituto Politecnico Nacional
Escuela Superior de Fisica y Matematicas
Licenciatura De Matematica Algoritmica
Fundamentos de inteligencia Artificial
Editor: Ortiz Ortiz Bosco
Titulo: Modelo de optimizador y costo
"""
#------------------------------------------
# 1)Dienyo de Modelo(entrada,salida,
# evaluacion multicapa)
# 2)Construccion de la funcion de optimizacion
# y la funcion de costo
# 3)Ciclo de entrenamiento
#   -Evaluacion=Calculo de la prediccion 
#    y costo
#   -Derivacion=Calculo de Gradientes
#   -Actualizacion de pesos
import torch
import torch.nn as nn
# Regresion Lineal
# f = w * x 
# Aqui : f = 2 * x
# 0)Muestras de entrenamiento, 
# Observe el tamanyo!
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
n_samples, n_features = X.shape
print(f'#muetras: {n_samples}, #caracteristicas: {n_features}')
# 0)Crea un muestra
X_test = torch.tensor([5], dtype=torch.float32)
# 1)Disenyo de Modelo,el modelo tiene que 
# implementar la evaluacion!
# Aqui se puede usar el modelo ya hecho 
# de torch
input_size = n_features
output_size = n_features
# El modelo puede ser llamado para X muestras
model = nn.Linear(input_size, output_size)
'''
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define diferent layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)
'''
print(f'Prediccon antes del entrenamiento: f(5) = {model(X_test).item():.3f}')
# 2)Costo y optimizacion
learning_rate = 0.01
n_iters = 100
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# 3)Ciclo de entrenamiento
for epoch in range(n_iters):
    # prediccion=Evaluacion con modelo
    y_predicted = model(X)
    # costo
    l = loss(Y, y_predicted)
    # calculo de gradientes=paso de derivacion
    l.backward()
    # actualizacion de pesos
    optimizer.step()
    # gradientes en zero despues del paso
    optimizer.zero_grad()
    if epoch % 10 == 0:
        # desempaquetacion de los parametros
        [w, b] = model.parameters() 
        print('iteracion', epoch+1, ': w = ', w[0][0].item(), 'costo = ', l)

print(f'Prediccion despues del entrenamiento: f(5) = {model(X_test).item():.3f}')
#******************************************
