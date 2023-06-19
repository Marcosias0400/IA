#******************************************
"""
Instituto Politecnico Nacional
Escuela Superior de fisica y Matematicas
Licenciatura de Matematica Algoritmica
Fundamentos de Inteligencia Artificial
Editor: Ortiz Ortiz Bosco
Titulo: Red Neuronal Convolucional
"""
#------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
#------------------------------------------
# Configuracion del dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#------------------------------------------
# Hyper-parametros
num_epochs = 5
batch_size = 4
learning_rate = 0.001
#------------------------------------------
# E conjunto tiene PILImage imagenes 
# en el rango [0, 1]. 
# TRansformaremos en imagenes 
# normalizadas en  [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#------------------------------------------
# CIFAR10: 60000 32x32 imagenes a color 
# en 10 clases, con 6000 imagenes por clase
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
def imshow(img):
    img = img / 2 + 0.5  # desnormalizar
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
# Tomar muestras aleatorias 
dataiter = iter(train_loader)
images, labels = next(dataiter)
# Mostar imagenes
imshow(torchvision.utils.make_grid(images))
#------------------------------------------
# Red Neuronal Convolucional
class ConvNet(nn.Module):
    #======================================
    # Constructor
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    #======================================
    # Evaluador
    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
        x = x.view(-1, 16 * 5 * 5)            # -> n, 400
        x = F.relu(self.fc1(x))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 10
        return x
model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # forma de origen: 
        # [4, 3, 32, 32] = 4, 3, 1024
        # Entrada: 3 canales de entrada,
        # 6 canales de salida, 
        # 5 tamanyo del nucleo
        images = images.to(device)
        labels = labels.to(device)
        # Paso de evaluacion
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Derivacion y optimizacion
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 2000 == 0:
            print (f'Iteracion [{epoch+1}/{num_epochs}], Paso [{i+1}/{n_total_steps}], Costo: {loss.item():.4f}')

print('Finalizacion de entrenamiento')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1
    acc = 100.0 * n_correct / n_samples
    print(f'Credibilidad de la red: {acc} %')
    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Credibilidad de la clase: {classes[i]}: {acc} %')
#******************************************
