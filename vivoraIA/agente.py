"""---------------------------------------------------------------------
Agente de vivora inteligente 20-Marzo-2023
------------------------------------------------------------------------
Traductor: Ortiz Ortiz Bosco
Materia: Fundamentos de Inteligencia Artificial
Instituto Politécnico Nacional
Escuela Superior de Física y Matemáticas
Licenciatura de Matemática Algorítmica
---------------------------------------------------------------------"""
#-----------------------------------------------------------------------
#MODULOS DE USO de python 
import torch #Este no esta en el python vanilla al menos de 
#3.6<=python x<= 3.11
import random
import numpy as np #Este no esta en el python vanilla al menos de 
#3.6<=python x<= 3.11
from collections import deque
#-----------------------------------------------------------------------
#MODULOS LOCALES
from juego import JuegoSnakeIA, Direccion, Punto
from modelo import Linear_QNet, QTrainer
from asistente import grafica
#-----------------------------------------------------------------------
#VARIABLES GLOBALES
MAX_MEMORIA=100_000
TAMANYO_DE_COLA=1000
LR=0.001
N_JUEGOS_AZAR=80
#-----------------------------------------------------------------------
#Clase Agente 
class Agente:
	#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	#Constructor
	# modelo: importa una red neuronal 'Linear_Qnet'
	# trainer: Es un 'Entrenador llamado desde modelo'
    def __init__(self):
        self.n_juegos=0
        self.epsilon=N_JUEGOS_AZAR # Alateoridad
        self.gamma=0.9 # Tasa de disminución
        self.memoria=deque(maxlen=MAX_MEMORIA) #Pila de memoria: popleft()
        self.modelo=Linear_QNet(11, 256, 3)
        self.entrenador=QTrainer(self.modelo, lr=LR, gamma=self.gamma)
	#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	#Estado del agente
    def obtener_estado(self,juego):
        cabeza=juego.snake[0]
        #===============================================================
        #Pixel de 20*20 pixeles es el tamaño de la cabeza
        punto_i=Punto(cabeza.x-20,cabeza.y)#i=izquierda
        punto_d=Punto(cabeza.x+20,cabeza.y)#d=derecha
        punto_s=Punto(cabeza.x,cabeza.y-20)#s=superior(arriba)
        punto_a=Punto(cabeza.x,cabeza.y+20)#a=abajo
        #===============================================================
        dir_i=juego.direccion==Direccion.IZQUIERDA
        dir_d=juego.direccion==Direccion.DERECHA
        dir_s=juego.direccion==Direccion.ARRIBA
        dir_a=juego.direccion==Direccion.ABAJO
		#===============================================================
        estado=[
            # Peligro enfrente
            (dir_d and juego.a_colisionado(punto_d)) or 
            (dir_i and juego.a_colisionado(punto_i)) or 
            (dir_s and juego.a_colisionado(punto_s)) or 
            (dir_a and juego.a_colisionado(punto_a)),
            # Peligro a la derecha
            (dir_s and juego.a_colisionado(punto_d)) or 
            (dir_a and juego.a_colisionado(punto_i)) or 
            (dir_i and juego.a_colisionado(punto_s)) or 
            (dir_d and juego.a_colisionado(punto_a)),
            # Peligro a la izquierda
            (dir_a and juego.a_colisionado(punto_d)) or 
            (dir_s and juego.a_colisionado(punto_i)) or 
            (dir_d and juego.a_colisionado(punto_s)) or 
            (dir_i and juego.a_colisionado(punto_a)),
            # Moverse a:
            dir_i, dir_d, dir_s, dir_a,
            # Ubicacion de la comida
            juego.comida.x<juego.cabeza.x,  # Comida a la izquierda
            juego.comida.x>juego.cabeza.x,  # Comida a la derecha
            juego.comida.y<juego.cabeza.y,  # Comida arriba
            juego.comida.y>juego.cabeza.y  # Comida abajo
            ]
		#El estado de un arreglo de 0 y 1 
        return np.array(estado, dtype=int)
	#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	#Funcion para asignar una cantidad de memoria
    def recordar(self,estado,accion,recompensa,sig_estado,fin):
        self.memoria.append((estado,accion,recompensa,sig_estado,fin)) 
        # popleft si el MAXIMO_MEMORIA  es superado
	#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	#funcion que entrena la memoria a largo plazo
    def memoria_largo_plazo(self):
        if len(self.memoria)>TAMANYO_DE_COLA:
			#Lista de tuplas
            mini_muestra=random.sample(self.memoria,TAMANYO_DE_COLA)
        else:
            mini_muestra=self.memoria
        #Extrae de un zip una lista de estados, recompenzas, etc.
        estados,acciones,recompenzas,sig_estados,finalizazciones=zip(*mini_muestra)
        self.entrenador.entrenar_paso(np.array(estados,dtype=np.float32),acciones,recompenzas,np.array(sig_estados,dtype=np.float32),finalizazciones)
	#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	#funcion que entrena a corto plazo
    def memoria_corto_plazo(self,estado,accion,recompensa,sig_estado,fin):
        self.entrenador.entrenar_paso(estado,accion,recompensa,sig_estado,fin)
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #Funcion que obtiene el estado
    def obtener_accion(self,estado):
        self.epsilon=N_JUEGOS_AZAR-self.n_juegos
        mov_final=[0,0,0]
        #Genera un numero aleatorio entre 0 y 200 y si es menor:
        if random.randint(0,200)<self.epsilon:
            movimiento=random.randint(0,2) #movimiento aleatorio
            mov_final[movimiento]=1 #La entrada k se hace 1
        else: #Si se rebasa la cantidad de juegos aleatorios entonces:
			#Hace una prediccion con su informacion para el siguiente 
			#movimiento
            estado0=torch.tensor(estado,dtype=torch.float)
            #Dado el estado(T11) obtiene una prediccion en un vector de R3
            prediccion=self.modelo(estado0)
            movimiento=torch.argmax(prediccion).item()
            mov_final[movimiento]=1
        return mov_final
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#-----------------------------------------------------------------------
def entrenar():
	#Parametros iniciales del juego
    grafica_puntajes=[]
    grafica_puntaje_medio=[]
    puntaje_total=0
    record=0
    agente=Agente()
    juego=JuegoSnakeIA()
    #===================================================================
    #Ciclo infinito
    while True:
		#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		#Lineas de codigo para configurar sus estados y acciones
        # Obtiene el estado anterior
        estado_viejo=agente.obtener_estado(juego)
        # Obtener movimiento
        movimiento_final=agente.obtener_accion(estado_viejo)
        # hacer movimiento y obtener nuevo estado
        recompensa,fin,puntaje=juego.paso_juego(movimiento_final)
        estado_nuevo= agente.obtener_estado(juego)
        # Entrenar memoria a corto plazo
        agente.memoria_corto_plazo(estado_viejo,movimiento_final,recompensa,estado_nuevo,fin)
        # Recordar lo ya hecho
        agente.recordar(estado_viejo,movimiento_final,recompensa,estado_nuevo,fin)
		#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		#Linea de codigo por si pierde el juego
        if fin:
            # Entrenar memoria a largo plazo, graficar resultado
            juego.reiniciar()
            agente.n_juegos+=1
            agente.memoria_largo_plazo()
            #Linea para reemplazar el record
            if puntaje>record: 
                record=puntaje
                agente.modelo.guardar()
            #Impresion de el record del juego junto con su record
            #Así como el número de juegos
            print('Juego: '+str(agente.n_juegos)+'; Puntaje: '+str(puntaje)+'; Record:'+str(record))
            grafica_puntajes.append(puntaje)
            puntaje_total+=puntaje
            puntaje_medio=puntaje_total/agente.n_juegos
            grafica_puntaje_medio.append(puntaje_medio)
            grafica(grafica_puntajes,grafica_puntaje_medio)
        #Cada que pierde el juego grafica el puntaje y su media
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#-----------------------------------------------------------------------
# Se inicia el programa en cuanto se llama el programa medio terminal
if __name__ == '__main__':
    entrenar()
#-----------------------------------------------------------------------
