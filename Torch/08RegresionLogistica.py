#******************************************
"""
Instituto Politecnico Ncional
Escuela Superior de Fisica y Matematicas
Licenciatura de Matematica Algoritmica
Fundamentos de Inteligencia Artificial
Editor: Ortiz Ortiz Bosco
Titulo: Regresion logistica
"""
#------------------------------------------
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#------------------------------------------
# 0)Preparacion de datos
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
n_samples, n_features = X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
# Escala
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)
# 1)Modelo
# Modelo lineal f = wx + b ,
# Sigmoide al final
class Model(nn.Module):
    #======================================
    # costructor
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
    #======================================
    # Evaluador
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
#------------------------------------------
model = Model(n_features)
#------------------------------------------
# 2)Costo y optimizacion
num_epochs = 200
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#------------------------------------------
# 3)Ciclo de entrenamiento
for epoch in range(num_epochs):
    # Paso de evaluacion y costo
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    # Paso de derivacion y actualizacion
    loss.backward()
    optimizer.step()
    # Actualizacion de pesos y gradiente cero
    optimizer.zero_grad()
    if (epoch+1) % 10 == 0:
        print(f'Iteracion: {epoch+1}, costo = {loss.item():.4f}')


with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'Precicion: {acc.item():.4f}')
#******************************************
