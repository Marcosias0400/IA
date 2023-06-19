#******************************************
"""
Instituto Politecnico Nacional
Escuela superior de fisica y Maatematicas
Licenciatura de Matematica Algoritmica
Fundamentos de Inteligencia Artificial
Editor: Ortiz Ortiz Bosco
Titulo: Caraga de datos
"""
#------------------------------------------
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
#------------------------------------------
# Calculo de gradiente etc. 
# no tan eficiente con todo los datos
# -> dividir el conjunto de datos por 
# segmentos
'''
# Ciclo de aprendizaje
for epoch in range(num_epochs):
    # loop over all batches
    for i in range(total_batches):
        batch_x, batch_y = ...
'''
# iteracion = una evalucacion y derivacion
# para todas las muestras 
# batch_size = Tamanyo de segmentos por
# por muestra
# number of iterations = numeor de pases, 
# cada paso (evalucaion+derivacion) 
# usando [tamanyo_segmento] al numero de 
# muestras
# e.g : 100 samples, batch_size=20 -> 100/20=5 iterations for 1 epoch
# --> CargadorDatos puede hacer el calculo
# por segmentos
# Implementar un conjunto de datos 
# personalizado:
# conjunto de datos inherencte 
# implementar  __init__ , __getitem__ , y
#  __len__
#------------------------------------------
class WineDataset(Dataset):
    #======================================
    # Contructor
    def __init__(self):
        # Descaraga e inicializa los datos,etc.
        # leer con numpy o pandas
        xy = np.loadtxt('./data/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]
        # la primera columna es el nombre
        # del conjunto de datos, el resto
        # es el resto de caracteristicas
        self.x_data = torch.from_numpy(xy[:, 1:]) # size [n_samples, n_features]
        self.y_data = torch.from_numpy(xy[:, [0]]) # size [n_samples, 1]
    #======================================
    # soporta acceso por medio de indices
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    # se puede usar len() para medir el
    # tamanyo del conjunto de datos
    def __len__(self):
        return self.n_samples
# Crear el conjunto de datos
dataset = WineDataset()
# tome los primeros datos y desempaque
first_data = dataset[0]
features, labels = first_data
print(features, labels)
# Cargue todos los datos con el cargador
# mezclador: mezclar los datos es bueno para 
# el entrenamiento
# num_workers: son la cantidad de subprocesos
# para trabajar las muestras
# !!! SI HAY UN ERROR, CONFIGURE num_workers A 0 !!!
train_loader = DataLoader(dataset=dataset,
                          batch_size=4,
                          shuffle=True,
                          num_workers=2)
# convertir el entrenador en un iterador
# para cargar las muestras 
dataiter = iter(train_loader)
data = next(dataiter)
features, labels = data
print(features, labels)
# Entrenamiento de practica
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print(total_samples, n_iterations)
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        # Aqui: 178 muestras, tamanyo 
        # de segmento = 4, 
        # n_iters=178/4=44.5 -> 45 iteraciones
        # Corre el proceso de entrenamieno
        if (i+1) % 5 == 0:
            print(f'Iteracion: {epoch+1}/{num_epochs}, Paso {i+1}/{n_iterations}| entradas {inputs.shape} | Caracteristica {labels.shape}')
# algunos conjuntos de datos estan disponibles
# en torchvision.datasets
# e.j. MNIST, Fashion-MNIST, CIFAR10, COCO
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=torchvision.transforms.ToTensor(),  
                                           download=True)

train_loader = DataLoader(dataset=train_dataset, 
                                           batch_size=3, 
                                           shuffle=True)
# Ver una muestra aleatoriamente 
dataiter = iter(train_loader)
data = next(dataiter)
inputs, targets = data
print(inputs.shape, targets.shape)
#******************************************
