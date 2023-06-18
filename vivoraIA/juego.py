"""---------------------------------------------------------------------
Definicion del juego 20-marzo-2023
------------------------------------------------------------------------
Traductor: Ortiz Ortiz Bosco
Materia: Fundamentos de Inteligencia Artificial
Instituto Politécnico Nacional
Escuela Superior de Física y Matemáticas
Licenciatura de Matemática Algorítmica
---------------------------------------------------------------------"""
#-----------------------------------------------------------------------
#MODULOS DE PYTHON A IMPORTAR
import pygame #Este no esta en el python vanilla al menos de 
#3.6<=python x<= 3.11
import random
from enum import Enum
from collections import namedtuple
import numpy as np #Este no esta en el python vanilla al menos de 
#3.6<=python x<= 3.11
#-----------------------------------------------------------------------
#Defincion del aspecto gráfico del juego basado en python
pygame.init() #Inicia el juego de python
font=pygame.font.Font('arial.ttf',25) #Definición de su fuente
#-----------------------------------------------------------------------
#Clase Direccion que retorna 4 opciones
class Direccion(Enum): #Es clase hija de Enum, un tipo de diccionario
    DERECHA=1
    IZQUIERDA=2
    ARRIBA=3
    ABAJO=4
#-----------------------------------------------------------------------
#Apartir de aqui se definen algunas variables globales al menos para el
#juego
#Define una tupla con nombre para poder ser llamda a posterior
Punto=namedtuple('Point', 'x, y') 
#COLORES PARA RGB
BLANCO=(255,255,255)
ROJO=(200,0,0)
AZUL1=(0,255,0)
AZUL2=(0,255,200)
NEGRO=(0,0,0)
#Definicion del tamño de bloque
TAM_BLOQUE=20
#Definicion de la velocidad a la que corre
VELOCIDAD=40
#-----------------------------------------------------------------------
#Clase JuegoSnakeIA
class JuegoSnakeIA: #Es su propia clase
	#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	#Constructor
    def __init__(self,w=640,h=480):
        self.w=w #Longitud de la pantalla de juego
        self.h=h #Altura de la pantalla de juego
        # Instanciacion del display
        self.display=pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        #Define el tiempo al que corre Snake
        self.clock=pygame.time.Clock()
        self.reiniciar()
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	# Reiniciar el juego
    def reiniciar(self):
        # Instanciacion de la primera dirección
        self.direccion=Direccion.DERECHA
        #Define la cabeza como en medio de la celda
        self.cabeza=Punto(self.w/2,self.h/2)
        #Lista con los atributos de la pocicion de la cabeza de la vivora
        self.snake=[self.cabeza,
                      Punto(self.cabeza.x-TAM_BLOQUE, self.cabeza.y),
                      Punto(self.cabeza.x-(2*TAM_BLOQUE), self.cabeza.y)]
        self.puntaje=0
        self.comida=None
        self._sembrar_comida()
        self.frame_iteration=0
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #Funcion que coloca comida
    def _sembrar_comida(self):
		#Genera coordenadas aleatorias para pocicionar la comida
        x=random.randint(0,(self.w-TAM_BLOQUE)//TAM_BLOQUE)*TAM_BLOQUE
        y=random.randint(0,(self.h-TAM_BLOQUE)//TAM_BLOQUE)*TAM_BLOQUE
        self.comida=Punto(x,y)
        if self.comida in self.snake:
            self._sembrar_comida()
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #DFuncion que actua sobre el siguiente paso de 'Snake'
    def paso_juego(self,accion):
        self.frame_iteration+=1
        # 1. colecta la accion del usuario (agente)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. Movimiento
        self._moverse(accion) # Actualizar la cabeza de la vivora
        self.snake.insert(0, self.cabeza)
        
        # 3. Ver si se perdio en el juego
        self.recompensa=0
        game_over=False
        if self.a_colisionado() or self.frame_iteration > 100*len(self.snake):
            game_over=True
            self.recompensa=-10
            return self.recompensa, game_over, self.puntaje
        # 4. colocar nueva comida o solo moverse
        if self.cabeza==self.comida:
            self.puntaje+=1
            self.recompensa+=10
            self._sembrar_comida()
        else:
            self.snake.pop()
        #Penalizar por tiempo
        if self.frame_iteration>20:
            self.recompensa-=int(0.001*self.frame_iteration)
        # 5. actualizar interfaz grafica y el tiempo
        self._update_ui()
        self.clock.tick(VELOCIDAD)
        # 6. retorna la recompenza, el perder y el puntaje
        return self.recompensa,game_over,self.puntaje
	#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	#Definición de la colicion de la serpiente
    def a_colisionado(self, pt=None):
        if pt is None:
            pt=self.cabeza
        # golpe en contra las paredes
        if pt.x>self.w-TAM_BLOQUE or pt.x<0 or pt.y>self.h-TAM_BLOQUE or pt.y<0:
            return True
        # golpe contra si mismo
        if pt in self.snake[1:]:
            return True
        return False
	#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	#Función qeu actualiza la interfaz gráfica
    def _update_ui(self):
        self.display.fill(NEGRO)
        for pt in self.snake:
            pygame.draw.rect(self.display,AZUL1,pygame.Rect(pt.x,pt.y,TAM_BLOQUE,TAM_BLOQUE))
            pygame.draw.rect(self.display,AZUL2,pygame.Rect(pt.x+4,pt.y+4,12,12))
        pygame.draw.rect(self.display, ROJO, pygame.Rect(self.comida.x,self.comida.y,TAM_BLOQUE,TAM_BLOQUE))
        texto=font.render("Puntaje: "+str(self.puntaje), True, BLANCO)
        self.display.blit(texto,[0, 0])
        pygame.display.flip()
	#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	#Función de movimiento
    def _moverse(self,accion):
		#Tupla de comportamiento de enteros de numpy
        # [recto,derecha,izquierda]
        clock_wise=[Direccion.DERECHA,Direccion.ABAJO,Direccion.IZQUIERDA,Direccion.ARRIBA]
        idx=clock_wise.index(self.direccion)
        #Cambio de direccion
        if np.array_equal(accion, [1, 0, 0]):
            nueva_dir=clock_wise[idx] # no cambia
        elif np.array_equal(accion, [0, 1, 0]):
            sig_idx=(idx+1)%4
            nueva_dir=clock_wise[sig_idx] # paso a la derecha d -> a -> i -> s
        else: # [0, 0, 1]
            sig_idx=(idx-1)%4
            nueva_dir=clock_wise[sig_idx] # paso a la izquierda d -> s -> i -> a
		#Nueva direccion
        self.direccion=nueva_dir
        x=self.cabeza.x
        y=self.cabeza.y
        if self.direccion==Direccion.DERECHA:
            x+=TAM_BLOQUE
        elif self.direccion==Direccion.IZQUIERDA:
            x-=TAM_BLOQUE
        elif self.direccion==Direccion.ABAJO:
            y+=TAM_BLOQUE
        elif self.direccion==Direccion.ARRIBA:
            y-=TAM_BLOQUE
        self.cabeza=Punto(x,y)
#-----------------------------------------------------------------------
