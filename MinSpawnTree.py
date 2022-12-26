from graph import Graph
import numpy as np
import random


def prim(graph: Graph, root=-1) -> Graph:
    """由图结构得到其最小生成树

    Parameters
    ----------
    graph : Graph
        图对象
    root : int, optional
        生成树的根节点, 缺省值为-1

    Returns
    -------
    Graph:
        图对象, 仅保留作为最小生成树的边
    """
    new_graph = Graph(graph.size)
    cur = []
    rest = graph.nodes.copy()
    if root == -1:
        root = random.randint(0, graph.size-1)
    cur.append(graph.nodes[root])
    sumdist = 0
    rest.remove(cur[0])
    while rest:
        shortest = (np.inf, None, None)
        for neighb in rest:
            for node in cur:
                dist = node.get_weight(neighb)
                if dist != -1 and dist < shortest[0]:  # no need to consider which ones are the real neighbours
                    shortest = (dist, node, neighb)
        cur.append(shortest[2])
        rest.remove(shortest[2])
        new_graph.nodes[shortest[1].index].addNeighbour(new_graph.nodes[shortest[2].index], shortest[0])
        sumdist += shortest[0]
    print('最短距离: '+str(sumdist))
    A = new_graph.getAdjMatrix()
    A = A + A.T
    new_graph = Graph(graph.size)
    new_graph.generate(A)
    return new_graph


if __name__ == '__main__':
    graph = Graph(7)
    A = np.array([[0, 50, 60, 0, 0, 0, 0],
                  [50, 0, 0, 65, 40, 0, 0],
                  [60, 0, 0, 52, 0, 0, 45],
                  [0, 65, 52, 0, 50, 30, 42],
                  [0, 40, 0, 50, 0, 70, 0],
                  [0, 0, 0, 30, 70, 0, 0],
                  [0, 0, 45, 40, 0, 0, 0]])
    graph.generate(A)
    new_graph = prim(graph)
    print(new_graph.getAdjMatrix())
