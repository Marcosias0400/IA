#******************************************
"""
Instituto Politecnico Nacional 
Escuela Superior de Fisica y Matematicas
Licenciatura en Matematica Algoritmica
Fundamentos de Inteligencia Artificial
Editor: Ortiz Ortiz Bosco
Titulo: Retropropagacion
"""
import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)
#------------------------------------------
# Este es el parametro a optimizar
#  -> requires_grad=True
w = torch.tensor(1.0, requires_grad=True)
# Evaluar para calcular el costo
y_predicted = w * x
loss = (y_predicted - y)**2
print(loss)
# backward pasa a calcular el gradiente
# dLoss/dw
loss.backward()
print(w.grad)
#------------------------------------------
# Actualiza pesos
# Pasa al la siguiente evaluacion y 
# derivacion...
# Continuar optimizando:
# Actualizar los pesos, Esat operacion no 
# deberia formar parte del calculo de 
# gradiente
with torch.no_grad():
    w -= 0.01 * w.grad
# No olvidar vaciar los calculos de gradiente
w.grad.zero_()
#------------------------------------------
# Pasa a la siguiente evaluacion y 
# derivacion...
#******************************************
