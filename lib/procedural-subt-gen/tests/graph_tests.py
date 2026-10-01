from subt_proc_gen.graph import Graph, Node, Edge, assert_unidirectionality
from subt_proc_gen.display_functions import plot_edges, plot_nodes
import random as rnd
import numpy as np
import pyvista as pv


def test1():
    g = Graph()
    n1 = Node((0, 0, 0))
    n2 = Node(np.random.random(3))
    n3 = Node([1, 2, 3])
    n4 = Node((1, 2, 3))


def test2():
    g = Graph()
    for _ in range(1000):
        g.add_node(Node(np.random.random(3) * 100))
    for _ in range(200):
        ni, nj = rnd.choices(tuple(g.nodes), k=2)
        g.connect(ni, nj)
    edges = tuple(g.edges)
    to_remove = set(list(set(rnd.choices(edges, k=50))))
    for n_edge, edge in enumerate(set(to_remove)):
        assert isinstance(edge, Edge)
        n1, n2 = tuple(edge.nodes)
        g.disconnect(n1, n2)
    assert_unidirectionality(g)
    plotter = pv.Plotter()
    plot_nodes(plotter, g.nodes)
    plot_edges(plotter, g.edges, color="r")
    plotter.show()


if __name__ == "__main__":
    test1()
    test2()
