from logging import basicConfig, INFO
from numpy import array, eye
from topolink import Graph

basicConfig(level=INFO)

L = array(
    [
        [2, 0, -1, -1, 0, 0, 0, 0, 0, 0],
        [0, 3, -1, 0, 0, -1, 0, -1, 0, 0],
        [-1, -1, 2, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 2, 0, 0, 0, 0, 0, -1],
        [0, 0, 0, 0, 1, 0, -1, 0, 0, 0],
        [0, -1, 0, 0, 0, 2, -1, 0, 0, 0],
        [0, 0, 0, 0, -1, -1, 2, 0, 0, 0],
        [0, -1, 0, 0, 0, 0, 0, 2, -1, 0],
        [0, 0, 0, 0, 0, 0, 0, -1, 1, 0],
        [0, 0, 0, -1, 0, 0, 0, 0, 0, 1],
    ]
)
W = eye(10) - L * 0.15

graph = Graph.from_mixing_matrix(W)
graph.deploy()
