#******************************************
"""
Instituto Politecnico Nacional
Escuela Superior de Fisica y Matematicas
Licenciatura en Matematica Algoritmica
Fundamentos de Inteligencia Artificial
Editor: Ortiz Ortiz Bosco
Titulo: Autograd
"""
import torch
#------------------------------------------
# La variable autograd provee una manera de 
# diferenciar automaticamente a los tensores  
# requires_grad = True -> 
# arrastra todas las operaciones del tensor  
x=torch.randn(3,requires_grad=True)
y=x+2
#------------------------------------------
# y fue creado como resultado de 
# una operacion de x, por lo que jala los 
# atributo de x
# grad_fn: refiere a una funcion que fue 
# creado el tensor 
print(x) 
# creado por el usuario -> grad_fn es None
print(y)
print(y.grad_fn)
#------------------------------------------
# Hacer mas operaciones sobre y
z=y*y*3
print(z)
z=z.mean()
print(z)
#------------------------------------------
# Calculemos los gradientes con 
# retropropagación
# Al terminar el computo del gradiente llama
# .backward() para llamar todos los 
# gradientes disponibles
# El gradiente de todos los tensores se 
# acumula en el atributo .grad
# Es la derivada parcial w.r.t. del tensor
z.backward()
print(x.grad) 
# dz/dx
#------------------------------------------
# Generalmente, torch.autograd es una 
# herramienta para calcular el Jacobiano 
# Calcula las derivadas parciales conforme 
# a la regla de la cadena 
# -----------------------------------------
# Modelo con una salida no escalar:
# Si la salida es un no escalar
# (mas de un elemento), necesitamos 
# especificar los argumenos para backguard() 
# Especificar el argumento 'gradient' 
# es decirle el tamanyo del tensor de salida.
# Necesario pra el producto vector-Jacobiano
x = torch.randn(3, requires_grad=True)
y = x * 2
for _ in range(10):
    y = y * 2
print(y)
print(y.shape)
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float32)
y.backward(v)
print(x.grad)
# -----------------------------------------
# quitar el arrastre de datos de un tensor:
# Por ejemplo, durante el ciclo de entenamiento
# cuando queremos actualizar los pesos
# entonces la operacion de actualizacion
# no deberia ser parte del calculo de
# gradiente  
# - x.requires_grad_(False)
# - x.detach()
# - ingrese como 'with torch.no_grad():'
# Cambia la etiqueta en el mismo momento
# .requires_grad_(...) 
a = torch.randn(2, 2)
print(a.requires_grad)
b = ((a * 3) / (a - 1))
print(b.grad_fn)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)
#-----------------------------------------
# copia el mismo vector pero con el 
# parametro grad desactivado, pra que no
# caldule el gradiente
# .detach(): 
a = torch.randn(2, 2, requires_grad=True)
print(a.requires_grad)
b = a.detach()
print(b.requires_grad)
#------------------------------------------
# Ingreselo 'with torch.no_grad():'
a = torch.randn(2, 2, requires_grad=True)
print(a.requires_grad)
with torch.no_grad():
    print((x ** 2).requires_grad)
#------------------------------------------
# backward() acumula el gradiente en el
# atributo .grad 
# Hay que ser cuidadosos con la
# optimizacion !!!
# Use .zero_() antes de otro paso de 
# optimizacion
weights = torch.ones(4, requires_grad=True)
for epoch in range(3):
    # Un ejemplo simple
    model_output = (weights*3).sum()
    model_output.backward()
    
    print(weights.grad)

    # Optimizando el modlo sin gradiente
    with torch.no_grad():
        weights -= 0.1 * weights.grad

    # Es importante modifica los pesos y
    # la salida resultante
    weights.grad.zero_()

print(weights)
print(model_output)
#------------------------------------------
# optimizador tiene el metodo zero_grad() 
# optimizer=torch.optim.SGD([weights],lr=0.1)
# Durante el entrenamiento:
# optimizer.step()
# optimizer.zero_grad()
#******************************************
