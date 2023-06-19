#******************************************
"""
Instituto Politecnico Nacional
Escuela Superior de Fisica y Matematicas
Licenciatura de Matematica Algoritmica
fundamentos de Inteligencia Artificial
Editor: Ortiz Ortiz Bosco
Titulo: Regresion Lineal
"""
#------------------------------------------
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
#------------------------------------------
# 0)Preparacion de datos
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
# Cambio a tensor Flotante
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)
n_samples, n_features = X.shape
#------------------------------------------
# 1)Modelo
# Modelo Lineal f = wx + b
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)
# 2)Costo y Optimizador
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  
# 3)Cilco de Entrenamiento
num_epochs = 200
for epoch in range(num_epochs):
    # Paso de Costo y evaluacion
    y_predicted = model(X)
    loss = criterion(y_predicted, y)
    # Derivacion y actualizacion
    loss.backward()
    optimizer.step()
    # gradiente a cero antes de otro paso
    optimizer.zero_grad()
    if (epoch+1) % 10 == 0:
        print(f'Iteracion: {epoch+1}, Costo = {loss.item():.4f}')
#------------------------------------------
# Graficacion
predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()
#******************************************
