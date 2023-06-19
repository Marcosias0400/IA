#******************************************
"""
Instituto Politecnico Nacional 
Escuela superior de Fisica y Matematcias
Licenciatura de Matematica Algoritmica
Fundamentos de inteligencia Artificial
Editor: Ortiz Ortiz Bosco
Titulo Tranformadores
"""
'''
Los tranformadores pueden ser aplicados a 
casi calquier tipo de dato para que pueda
procesarlos torch
Lista completa de transofrmadores ya hechos: 
https://pytorch.org/docs/stable/torchvision/transforms.html
-------------------------------------------
En imagenes:
CenterCrop, Grayscale, Pad, RandomAffine
RandomCrop, RandomHorizontalFlip, RandomRotation
Resize, Scale
-------------------------------------------
En tensores:
LinearTransformation, Normalize, RandomErasing
-------------------------------------------
Conversion:
ToPILImage: para tensores o ndrarray
ToTensor : para numpy.ndarray o PILImage
-------------------------------------------
Genericos:
Use Lambda personalizado
Escriba su propia clase
-------------------------------------------
Componga multiples Transformaciones:
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])
'''
#------------------------------------------
import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np
class WineDataset(Dataset):
    #======================================
    # Contructor de conjunto de datos
    def __init__(self, transform=None):
        xy = np.loadtxt('./data/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]
        # note aun no se convierte a tensor
        self.x_data = xy[:, 1:]
        self.y_data = xy[:, [0]]
        self.transform = transform
    #======================================
    # Indice
    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[index]
        if self.transform:
            sample = self.transform(sample)
        return sample
    #======================================
    # Tamanyo del conjunto de datos 
    def __len__(self):
        return self.n_samples
#------------------------------------------
# Transformaciones personalizadas
# implementa __call__(self, sample)
class ToTensor:
    # convierte ndarrays a tensores
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)
#------------------------------------------
class MulTransform:
    # multiplica la cantidad de salidas dado
    # un factor de forma
    def __init__(self, factor):
        self.factor = factor
    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets
print('Without Transform')
dataset = WineDataset()
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)
print('\nWith Tensor Transform')
dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)
print('\nWith Tensor and Multiplication Transform')
composed = torchvision.transforms.Compose([ToTensor(), MulTransform(4)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)
#******************************************
