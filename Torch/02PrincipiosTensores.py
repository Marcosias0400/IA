#********************************************
"""
Instituto Politecnico Nacional
Escuela Superior de Fisica y Matematicas
Licenciatura de Matematica Algoritmica
Fundamentos de Inteligencia Artificial
Editor: Ortiz Ortiz Bosco
Titulo: Principios de Torch
"""
#--------------------------------------------
# Modulo 'pytorch'
import torch
#--------------------------------------------
"""
Todo en pytorch es en operaciones tensoriales.
 Un tensor puede ser de diferentes dimensiones
 Puede ser de dim>=1
"""
#--------------------------------------------
# escalares, vectores, matrices, tensores
# torch.empty(size): Tensor no inicializado
x = torch.empty(1) # Escalar
print(x)
#--------------------------------------------
x = torch.empty(3) # vector, 1D
print(x)
#--------------------------------------------
# matriz, 2 dimensiones, M_(2,3)
x = torch.empty(2,3) 
print(x)
#--------------------------------------------
# tensor, 3 dimensiones, 
x = torch.empty(2,2,3) 
#--------------------------------------------
# tensor, 4 dimensiones
x = torch.empty(2,2,2,3) 
print(x)
#--------------------------------------------
# torch.rand(size): numeros aleatorios [0,1]
x = torch.rand(5, 3)
print(x)
#--------------------------------------------
# torch.zeros(size), llena el tensor de  0
# torch.ones(size), llena el tensor de  1
x = torch.zeros(5, 3)
print(x)
#--------------------------------------------
# Verifica el tamanyo
print(x.size())
#--------------------------------------------
# verifica el tipo de dato
print(x.dtype)
#--------------------------------------------
# especifica el tipo de dto, float32 defecto
x = torch.zeros(5, 3, dtype=torch.float16)
print(x)
#--------------------------------------------
# verifica el tipo
print(x.dtype)
#--------------------------------------------
# Tensor construido desde los datos
x = torch.tensor([5.5, 3])
print(x.size())
#--------------------------------------------
""" Argumento 'requires_grad' 
Le dice a pytorch si necesita calcular el
gradiente despues en los procesos de 
optimizacion i.e. esta variable en el modelo
que quieres optimizar
"""
x = torch.tensor([5.5, 3], requires_grad=True)
#--------------------------------------------
# Operaciones
y = torch.rand(2, 2)
x = torch.rand(2, 2)
# Adicion elemental
z = x + y
"""forma alternativa 'torch.add(x,y)'
 en lugar de la adicion, cualquier operacion
 que se le hagan se modifica el operador 
 y.add_(x)"""

# substraccion
z = x - y
z = torch.sub(x, y)

# multiplicacion
z = x * y
z = torch.mul(x,y)

# division
z = x / y
z = torch.div(x,y)
#--------------------------------------------
# Vector aleatorio  R^2 x R^3 
x = torch.rand(5,3)
print(x)
print(x[:, 0]) # Todas la filas, Columna 0
print(x[1, :]) # Fila 1, Todas las columnas
print(x[1,1]) # elemento (1,1)
#-------------------------------------------
# Toma el elemento actual en (1,1)
print(x[1,1].item())
#------------------------------------------
# Redimensiona con torch.view()
x = torch.randn(4, 4)
y = x.view(16)
#------------------------------------------
# La dim "-1" es inferida de las otras dimensiones 
z = x.view(-1, 8)   
#------------------------------------------
print(x.size(), y.size(), z.size())
#------------------------------------------
# Numpy
# La conversion de tensores de torch a 
# arreglos de numpy es muy sencillo
a = torch.ones(5)
print(a)

# torch a numpy con numpy()
b = a.numpy()
print(b)
print(type(b))
#------------------------------------------
# Advertencia: si el tensor esta en el CPU
# (no en el GPU) ambos objetos estaran 
# guardados en la misma locacion de 
# memoria,entonces cargar uno, modiicara 
# el otro retroactivamente
a.add_(1)
print(a)
print(b)
#------------------------------------------
# numpy a torch con .from_numpy(x)
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
print(a)
print(b)
#------------------------------------------
# otra vez, cuidado con las modificaciones
a += 1
print(a)
print(b)
#------------------------------------------
# Por defecto todos los tensores son 
# creados en el CPU pero pueden ser 
# relocalizados al GPU solo si este esta 
# dispoible
if torch.cuda.is_available():
    # un dispositivo CUDA 
    device = torch.device("cuda")          
    # crear directamente el tendor en el GPU
    y = torch.ones_like(x, device=device)  
    # o solo ua caracteres "".to("cuda")""
    x = x.to(device)                       
    z = x + y
    # no es posible por que numpy no puede
    # gestionar tensores en el GPU
    # z = z.numpy() 
    # Mover al GPU otra vez
    z.to("cpu")       
    # ".to" Tambien puede cambiar el tipo
    # de numpy tambien 
    # z = z.numpy()
#******************************************
