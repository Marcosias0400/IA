#******************************************
"""
Instituto Politecnico Nacional
Escuela Superior de Fisica y Matemticas
Licenciatura de Matematica Algoritmica
Fundamentos de Inteligencia Artificial
Editor: Ortiz Ortiz Bosco
Titulo: PSO
"""
#------------------------------------------
# modulos y objetos importados
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from particle import Particle
#------------------------------------------
# Clase de optimizacion usando enjambres
# de particulas
class PSOClusteringSwarm:
    #======================================
    # Constructor del enjambre
    def __init__(self, n_clusters: int, n_particles: int, data: np.ndarray, hybrid=True, w=0.72, c1=1.49, c2=1.49):
        #++++++++++++++++++++++++++++++++++
        # Inicializa el enjambre
        # :param n_clusters: numero de agru-
        #                    pamientos
        # :param n_particles: numero de par-
        #                     ticulas
        # :param data: (numero de puntos x 
        #              dimensiones)
        # :param hybrid: bool : si se usan
        #                      k promedios
        # :param w:
        # :param c1:
        # :param c2:
        self.n_clusters = n_clusters
        self.n_particles = n_particles
        self.data = data
        self.particles = []
        #++++++++++++++++++++++++++++++++++
        # Para guardar el mejor global
        self.gb_pos = None
        self.gb_val = np.inf
        #++++++++++++++++++++++++++++++++++
        # Mejor Agrupamiento global
        # para cada dato contiene el numero
        # de agrupamiento
        self.gb_clustering = None
        self._generate_particles(hybrid, w, c1, c2)
    def _print_initial(self, iteration, plot):
        print('*** Initialing swarm with', self.n_particles, 'PARTICLES, ', self.n_clusters, 'CLUSTERS with', iteration,
              'MAX ITERATIONS and with PLOT =', plot, '***')
        print('Data=', self.data.shape[0], 'points in', self.data.shape[1], 'dimensions')
    #======================================
    # Genera particulas con k agrupamientos 
    # y puntos en t dimensiones 
    def _generate_particles(self, hybrid: bool, w: float, c1: float, c2: float):
        for i in range(self.n_particles):
            particle = Particle(n_clusters=self.n_clusters, data=self.data, use_kmeans=hybrid, w=w, c1=c1, c2=c2)
            self.particles.append(particle)
    def update_gb(self, particle):
        if particle.pb_val < self.gb_val:
            self.gb_val = particle.pb_val
            self.gb_pos = particle.pb_pos.copy()
            self.gb_clustering = particle.pb_clustering.copy()
    #======================================
    # :param plot: en True graficara los
    #              mejores agrupamientos
    #              globales
    # :param tieration: nuemro de iteraciones
    #                   maximas
    # :retorna: (mejor agrupamieno, grafica) 
    def start(self, iteration=1000, plot=False) -> Tuple[np.ndarray, float]:
        self._print_initial(iteration, plot)
        progress = []
        #++++++++++++++++++++++++++++++++++
        # Itera hasta la maxima iteracion 
        for i in range(iteration):
            if i % 200 == 0:
                clusters = self.gb_clustering
                print('iteration', i, 'GB =', self.gb_val)
                print('best clusters so far = ', clusters)
                if plot:
                    centroids = self.gb_pos
                    if clusters is not None:
                        plt.scatter(self.data[:, 0], self.data[:, 1], c=clusters, cmap='viridis')
                        plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha=0.5)
                        plt.show()
                    else:  # if there is no clusters yet ( iteration = 0 ) plot the data with no clusters
                        plt.scatter(self.data[:, 0], self.data[:, 1])
                        plt.show()

            for particle in self.particles:
                particle.update_pb(data=self.data)
                self.update_gb(particle=particle)

            for particle in self.particles:
                particle.move_centroids(gb_pos=self.gb_pos)
            progress.append([self.gb_pos, self.gb_clustering, self.gb_val])

        print('Finished!')
        return self.gb_clustering, self.gb_val
#******************************************
