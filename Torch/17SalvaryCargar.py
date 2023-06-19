#******************************************
"""
Instituto Politecnico Nacional
Escuela superior de Fisica y Matematicas
Licenciatura de Matematica Algoritmica
Fundamentos de Inteligencia Artificial
Editor: Ortiz Ortiz Bosco
Titulo: Guardar y Cargar
"""
#------------------------------------------
import torch
import torch.nn as nn
#------------------------------------------
''' 3 diferentes metodos para cargar:
 - torch.save(arg, PATH) 
   puede ser modelo, tensor, o diccionario
 - torch.load(PATH)
 - torch.load_state_dict(arg)
'''

''' 2 formas diferentes formas de salvar
# 1)la forma perezosa: guardar tod el modelo
torch.save(model, PATH)
# La clase del modelo tiene que se guardado 
# en algun lado
model = torch.load(PATH)
model.eval()
# 2)forma recomendada: salvar solo state_dict
torch.save(model.state_dict(), PATH)
# modelo tiene que ser creados otra vez
# con nuevos parametros
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
'''
class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
model = Model(n_input_features=6)
# entrenar el modelo...
####################guardar todo######################################
for param in model.parameters():
    print(param)
# salvar y cargar el modelo
FILE = "model.pth"
torch.save(model, FILE)
loaded_model = torch.load(FILE)
loaded_model.eval()
for param in loaded_model.parameters():
    print(param)
######salvar solo el diccionario###########
############ de estados ###################
FILE = "model.pth"
torch.save(model.state_dict(), FILE)
print(model.state_dict())
loaded_model = Model(n_input_features=6)
loaded_model.load_state_dict(torch.load(FILE)) # it takes the loaded dictionary, not the path file itself
loaded_model.eval()
print(loaded_model.state_dict())
########carga el punto guardado###############
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
checkpoint = {
"epoch": 90,
"model_state": model.state_dict(),
"optim_state": optimizer.state_dict()
}
print(optimizer.state_dict())
FILE = "checkpoint.pth"
torch.save(checkpoint, FILE)
model = Model(n_input_features=6)
optimizer = torch.optim.SGD(model.parameters(), lr=0)
checkpoint = torch.load(FILE)
model.load_state_dict(checkpoint['model_state'])
optimizer.load_state_dict(checkpoint['optim_state'])
epoch = checkpoint['epoch']
model.eval()
# - o -
# model.train()
print(optimizer.state_dict())
# Recordatorio que se puede llamar 
# model.eval() con todo y particion de datos 
# to evaluation mode before running inference. Failing to do this will yield 
# inconsistent inference results. If you wish to resuming training, 
# call model.train() to ensure these layers are in training mode.

""" Salvar en GPU/CPU 
# 1) Salvar en  GPU, Cargar en  CPU
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), PATH)
device = torch.device('cpu')
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))
# 2) Salvar en GPU, Cargar en GPU
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), PATH)
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.to(device)
# Nota:Asegurate se usar 
# .to(torch.device('cuda'))  
# en todas las entradas usadas
# 3) Salvar en CPU, cargar en GPU
torch.save(model.state_dict(), PATH)
device = torch.device("cuda")
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # Choose whatever GPU device number you want
model.to(device)
# esto lo carga en el GPU. 
# Asegurese de usar  
# model.to(torch.device('cuda')) para convertir
# los tensores de torch a tensores de CUDA
"""
#******************************************

