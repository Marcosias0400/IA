#******************************************
"""
Isntituto Politcnico Nacional
Escuela Superiror de fisica y Matematicas
Licenciatura de Matematica Algorimmica
Fundamentos de Inteligencia Artificial
Editor: Ortiz Ortiz Bosco
Titulo: Agrupamiento usando conjuntos de particulas
"""
#------------------------------------------
# Modulos Importados
import pandas as pd
import numpy as np
from pso_clustering import PSOClusteringSwarm
#------------------------------------------
plot = True
#------------------------------------------
# Lectura de hoja de datos (usando pandas)
data_points = pd.read_csv('iris.txt', sep=',', header=None)
#------------------------------------------
# Pasar columna 4 (comienza en 0) a un
# a un arreglo de numpy
clusters = data_points[4].values
#------------------------------------------
# Remover la columna 4 usando el metodo drop
data_points = data_points.drop([4], axis=1)
#------------------------------------------
# Usar columnas 0 y 1 para graficar en 2D 
if plot:
    data_points = data_points[[0, 1]]
#------------------------------------------
# convetir lista a arreglo de numpy 
data_points = data_points.values
#------------------------------------------
# AGORITMO PSO-CLUSTERING
pso = PSOClusteringSwarm(n_clusters=3, n_particles=10, data=data_points, hybrid=True)
pso.start(iteration=1000, plot=plot)
# Mapeo de colores asociados a cada cluster
mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
clusters = np.array([mapping[x] for x in clusters])
print('Actual classes = ', clusters)
#******************************************
