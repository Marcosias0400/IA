#******************************************
"""
Instituto Politecnico Nacional
Escuela superior de Fisica y Matematicas
Licenciatura de Matematica Algoritmica
fundamentos de Inteligencia Artificial
Editor: Ortiz Ortiz Bosco
Titulo: Funcions de Activacion
"""
# salida = w*x + b
# salida = activation_function(salida)
#------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
#------------------------------------------
x = torch.tensor([-1.0, 1.0, 2.0, 3.0])
#------------------------------------------
# sofmax
output = torch.softmax(x, dim=0)
print(output)
sm = nn.Softmax(dim=0)
output = sm(x)
print(output)
#------------------------------------------
# sigmoide 
output = torch.sigmoid(x)
print(output)
s = nn.Sigmoid()
output = s(x)
print(output)
#------------------------------------------
#Tangente hiperbolica
output = torch.tanh(x)
print(output)
t = nn.Tanh()
output = t(x)
print(output)
#------------------------------------------
# Regresion lineal
output = torch.relu(x)
print(output)
relu = nn.ReLU()
output = relu(x)
print(output)
#------------------------------------------
# Regresion lineal con fugas
output = F.leaky_relu(x)
print(output)
lrelu = nn.LeakyReLU()
output = lrelu(x)
print(output)
#------------------------------------------
#nn.ReLU() crea un nn.Module tal que se puede
# agregar e.j. un modelo nn.Sequential.
#torch.relu es una funcion que solo llama a
# a la red neuronal lineal
#entonces sea lo que sea agregado e.j.
# como un evaluador personalizado.
#------------------------------------------
# opcion 1 crear modulos nn 
class NeuralNet(nn.Module):
    #======================================
    # Contructor de red Neuronal con
    # con modulo agregado
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    #======================================
    # Evaluador
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out
#------------------------------------------
# opcion 2 (usar funciones de activacion
# en el paso de evaluacion)
class NeuralNet(nn.Module):
    #======================================
    # constructor
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
    #======================================
    # Evaluador con funcion de activacion
    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        return out
#******************************************
