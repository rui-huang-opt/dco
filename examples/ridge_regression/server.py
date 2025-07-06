from logging import basicConfig, INFO
from topolink import Graph

basicConfig(level=INFO)

nodes = ["1", "2", "3", "4"]
edges = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "1")]

graph = Graph(nodes, edges)
graph.deploy()
