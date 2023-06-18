#******************************************
"""
Instituto Politecnico Nacional 
Escuela Superior de Fisica y Matematicas
Licenciatura de Matematica Algoritmica
FUNDAMENTOS DE INTELIGENCIA ARTIFICIAL
Editor: Ortiz Ortiz Bosco
titulo: Laberinto
"""
#------------------------------------------
# Importacion de objetos definidos en 
# pyamaze
from pyamaze import maze,agent
#------------------------------------------
# Crear Laberinto, (x,y): posicion de la
# meta
m=maze(25,40)
m.CreateMaze(x=1,y=1)
#------------------------------------------
# Ubicacion del agente del laberinto
a=agent(m,footprints=True,filled=True)
#------------------------------------------
# Graficar trayectoria del agente
m.tracePath({a:m.path},delay=25)
m.run()
#******************************************
