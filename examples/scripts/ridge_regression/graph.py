import numpy as np
from logging import basicConfig, INFO
from topolink import Graph

basicConfig(level=INFO)

L = np.array([[2, -1, 0, -1], [-1, 2, -1, 0], [0, -1, 2, -1], [-1, 0, -1, 2]])
W = np.eye(4) - L * 0.2

graph = Graph.from_mixing_matrix(W)

graph.deploy()
