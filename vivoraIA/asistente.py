"""------------------------------------------------------------------------
Definicion del graficador, denotado como 'asistente' 20-marzo-2023
---------------------------------------------------------------------------
Traductor: Ortiz Ortiz Bosco
Materia: Fundamentos de Inteligencia Artificial
Instituto Politécnico Nacional
Escuela Superior de Física y Matemáticas
Licenciatura de Matemática Algorítmica
-------------------------------------------------------------------------"""
#---------------------------------------------------------------------------
#MODULOS DE PYTHON A IMPORTAR, ninguno se encuentra en python vanilla 
# al menos 3.6<=python x<= 3.11
import matplotlib.pyplot as plt
from IPython import display
#---------------------------------------------------------------------------
#plt.ion()
# Definición de una grafica que cambia con respecto al tiempo
def grafica(puntaje, puntaje_medio):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('entrenando...')
    plt.xlabel('# de Juegos')
    plt.ylabel('Puntaje')
    plt.plot(puntaje)
    plt.plot(puntaje_medio)
    plt.ylim(ymin=0)
    plt.text(len(puntaje)-1,puntaje[-1],str(puntaje[-1]))
    plt.text(len(puntaje_medio)-1,puntaje_medio[-1],str(puntaje_medio[-1]))
    plt.show(block=False)
    plt.pause(.1)
#---------------------------------------------------------------------------
