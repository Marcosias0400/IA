#******************************************
"""
Instituto Politecnico Nacional
Escuela Superior de fisica y Matematicas
Licenciatura de Matematica Algoritmica
fundamentos de Inteligencia Artifical
Editor: Ortiz Ortiz Bosco
Titulo: alimentacion hacia adelante
"""
#------------------------------------------
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
#------------------------------------------
# Configuracion del dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#------------------------------------------
# Hyper-parametros
input_size = 784 # 28x28
hidden_size = 500 
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001
#------------------------------------------
# Conjunto de datos MNIST 
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transforms.ToTensor())
#------------------------------------------
# Cargador de datos
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)
examples = iter(test_loader)
example_data, example_targets = next(examples)
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(example_data[i][0], cmap='gray')
plt.show()
#-------------------------------------------
# Red neuronal totalmente conectada
# con una capa oculta
class NeuralNet(nn.Module):
    #======================================
    # Constructor
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)  
    #======================================
    # Evaluador
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # sin activacion ni softmax al final
        return out
#------------------------------------------
model = NeuralNet(input_size, hidden_size, num_classes).to(device)
# Optimizador y Costo
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
# Entrenando el modelo
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # Tamanyo original: [100, 1, 28, 28]
        # Reajuste: [100, 784]
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        # Paso de evaluacion
        outputs = model(images)
        loss = criterion(outputs, labels)
        # DErivacion y optimizacion
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print (f'Iteracion [{epoch+1}/{num_epochs}], Paso [{i+1}/{n_total_steps}], Costo: {loss.item():.4f}')
# Prueba del modelo
# En la fase de prueba, no se requiere calculo
# de gradiente (por eficiencia de memoria)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returna (valor ,indice)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
    acc = 100.0 * n_correct / n_samples
    print(f'Porcentaje de credibilidad en imagenes: {acc} %')
#******************************************
