"""---------------------------------------------------------------------
Definicion del modelo de inteligencia artificial 20-Marzo-2023
------------------------------------------------------------------------
Traductor: Ortiz Ortiz Bosco
Materia: Fundamentos de Inteligencia Artificial
Instituto Politécnico Nacional
Escuela Superior de Física y Matemáticas
Licenciatura de Matemática Algorítmica
---------------------------------------------------------------------"""
#-----------------------------------------------------------------------
#MODULOS A IMPORTAR DE PYTHON torch se encuentra en python vanilla 
# al menos 3.6<=python x<= 3.11
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#-----------------------------------------------------------------------
#Este si viene en python vanilla
import os
#-----------------------------------------------------------------------
#Red neuronal lineal Q
class Linear_QNet(nn.Module):
	#Constructor
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.linear1=nn.Linear(input_size,hidden_size)
        self.linear2=nn.Linear(hidden_size,output_size)
    #Predicción
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    #Salvar lo hecho
    def guardar(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name=os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
#-----------------------------------------------------------------------
# Aprendizaje optimo; Q es de Quality <------> calidad
class QTrainer:
	#Constructor: con un modelo a optimizar y un valor de gamma
    def __init__(self,modelo,lr,gamma):
        self.lr=lr
        self.gamma=gamma
        self.modelo=modelo
        self.optimizador=optim.Adam(modelo.parameters(),lr=self.lr)
        self.criterio=nn.MSELoss() #Minímos cuadrados
	#Entrenar por paso dado, ademas de convertir a tensores los vectores
    def entrenar_paso(self,estado,accion,recompensa,sig_estado,fin):
        estado=torch.tensor(estado, dtype=torch.float)
        sig_estado=torch.tensor(sig_estado,dtype=torch.float)
        accion=torch.tensor(accion, dtype=torch.long)
        recompensa=torch.tensor(recompensa,dtype=torch.float)
        # (n, x)
        if len(estado.shape)==1:
            # (1, x)
            estado=torch.unsqueeze(estado,0)
            sig_estado=torch.unsqueeze(sig_estado,0)
            accion=torch.unsqueeze(accion,0)
            recompensa=torch.unsqueeze(recompensa,0)
            fin=(fin,)
        # 1: prediccion de lo valores Q con el estado actual
        prediccion=self.modelo(estado)
        objetivo=prediccion.clone()
        for idx in range(len(fin)):
            Q_nueva=recompensa[idx]
            if not fin[idx]:
				#Ecuacion de Bellman incluida para optimizar el modelo
                Q_nueva=recompensa[idx]+self.gamma*torch.max(self.modelo(sig_estado[idx]))
            objetivo[idx][torch.argmax(accion[idx]).item()]=Q_nueva
    
        # 2: Q_nueva = r + y * max(prediccion del Q valor siguiente) 
        # -> solo hacer esto si no a finalizado
        # preds[argmax(action)] = Q_new
        self.optimizador.zero_grad()
        perdida=self.criterio(objetivo,prediccion)
        #Calculo del gradiente
        perdida.backward()
        #Optimizacion, pasar al siguiente mínimo
        self.optimizador.step()
#------------------------------------------------------------------------
