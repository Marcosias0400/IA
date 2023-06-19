#******************************************
"""
Instituto Politecnico Nacional
Escuela Superiror deFisica y Matematicas
Licenciatura de Matematica Algoritmica
Fundamentos de Inteligancia Artificial
Editor: Ortiz Ortiz Bosco
Titulo: Entropia cruzada y softmax
"""
#------------------------------------------
import torch
import torch.nn as nn
import numpy as np
#------------------------------------------
#        -> 2.0              -> 0.65  
# Lineal -> 1.0  -> Softmax  -> 0.25   -> EntropiaCruzada(y, y_hat)
#        -> 0.1              -> 0.1                   
#     puntaje(logits)      Proabailidades
#                           sum = 1.0
#------------------------------------------
# Softmax aplica LA fncion exponencial
# y normaliza dividiendo la suma de todas 
# estas -> comprime la salida entre 0 y 1 
# para que sean probabilidades
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print('softmax numpy:', outputs)
x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0) # along values along first axis
print('softmax torch:', outputs)
# Entropia Cruzada o costo cruzado son
# las medidas de salida en torno a una  
# probabilidad entre 0 y 1 
# -> la probabilidad diverge entre menor sea
# su recurrencia
def cross_entropy(actual, predicted):
    EPS = 1e-15
    predicted = np.clip(predicted, EPS, 1 - EPS)
    loss = -np.sum(actual * np.log(predicted))
    return loss # / float(predicted.shape[0])
# y tiene que ser codificado inmediatemanete
# si clase de 0: [1 0 0]
# si clase de 1: [0 1 0]
# si clase de 2: [0 0 1]
Y = np.array([1, 0, 0])
Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)
print(f'Loss1 numpy: {l1:.4f}')
print(f'Loss2 numpy: {l2:.4f}')

# CrossEntropyLoss en PyTorch (aplica Softmax)
# nn.LogSoftmax + nn.NLLLoss
# NLLLoss = perdida negativa logaritmica
loss = nn.CrossEntropyLoss()
# loss(entrada, objetivo)
# el objetivo es tener nSamples = 1
# cada elemnto puede ser : 0, 1, or 2
# Y (=obejtivo) contiene la clase no codificada
Y = torch.tensor([0])
# el tamanyo de las entradas son 
#  nSamples x nClasses = 1 x 3
# y_pred (=entrada) debe ser crudo, 
# sin modificar de alguna manera
Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]])
l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)
print(f'PyTorch Loss1: {l1.item():.4f}')
print(f'PyTorch Loss2: {l2.item():.4f}')
# Predicciones
_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print(f'Actual class: {Y.item()}, Y_pred1: {predictions1.item()}, Y_pred2: {predictions2.item()}')
# Permite el calculo de perdidas en multiples
# muestras
# objetivos de tamanyo nBatch = 3
# Cada elemento puede ser de clase: 0, 1, or 2
Y = torch.tensor([2, 0, 1])
# Cada entrada es de tamanyo
# nBatch x nClasses = 3 x 3
# Y_pred es logistico (no softmax)
Y_pred_good = torch.tensor(
    [[0.1, 0.2, 3.9], # prediccion clase 2
    [1.2, 0.1, 0.3], # prediccion clase 0
    [0.3, 2.2, 0.2]]) # prediccion clase 1
Y_pred_bad = torch.tensor(
    [[0.9, 0.2, 0.1],
    [0.1, 0.3, 1.5],
    [1.2, 0.2, 0.5]])
l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)
print(f'Batch Loss1:  {l1.item():.4f}')
print(f'Batch Loss2: {l2.item():.4f}')
# Predicciones
_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print(f'Actual class: {Y}, Y_pred1: {predictions1}, Y_pred2: {predictions2}')
#------------------------------------------
# Clasificacion Binaria
class NeuralNet1(nn.Module):
    #======================================
    # Contructor de una red neuronal
    def __init__(self, input_size, hidden_size):
        super(NeuralNet1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)  
    #======================================
    # Evaluador
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # sigmoide al final
        y_pred = torch.sigmoid(out)
        return y_pred
#------------------------------------------
model = NeuralNet1(input_size=28*28, hidden_size=5)
criterion = nn.BCELoss()
#------------------------------------------
# Problema Multiclase
class NeuralNet2(nn.Module):
    #======================================
    # Constructor de Red Neuronal con 
    # multiples clases
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)  
    #======================================
    # Evaluador 
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # sin softmax al final
        return out
#------------------------------------------
model = NeuralNet2(input_size=28*28, hidden_size=5, num_classes=3)
criterion = nn.CrossEntropyLoss()  
# (applica Softmax)
#******************************************

