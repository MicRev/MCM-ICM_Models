import numpy as np


class Node:
    def __init__(self, index):
        self.neighbours = list()
        self.weight = list()
        self.previous = None
        self.index = index

    def addNeighbour(self, node, weight: int):
        self.neighbours.append(node)
        self.weight.append(weight)

    def getWeight(self, node):
        if node not in self.neighbours:
            return -1
        idx = self.neighbours.index(node)
        return self.weight[idx]


class Graph:
    def __init__(self, size):
        self.size = size
        self.nodes = [Node(i) for i in range(size)]

    def generate(self, adj_matrix: np.array):
        """由邻接矩阵生成图结构

        Args:
            adj_matrix (np.array): 邻接矩阵
        """
        for idx, node in enumerate(self.nodes):
            row = adj_matrix[idx]
            for jdx, w in enumerate(row):
                if w != 0:
                    node.addNeighbour(self.nodes[jdx], w)

    def getAdjMatrix(self):
        """获得图的邻接矩阵

        Returns:
            np.array: 图对应的邻接矩阵
        """
        adj_mat = []
        for node in self.nodes:
            row = []
            for i in range(self.size):
                if self.nodes[i] in node.neighbours:
                    row.append(node.getWeight(self.nodes[i]))
                else:
                    row.append(0)
            adj_mat.append(row)
        return np.array(adj_mat)

    def clear(self):
        for node in self.nodes:
            node.previous = None


if __name__ == '__main__':
    graph = Graph(9)
    A = np.array([[0, 4, 0, 0, 0, 0, 0, 8, 0],
                  [4, 0, 8, 0, 0, 0, 0, 11, 0],
                  [0, 8, 0, 7, 0, 4, 0, 0, 2],
                  [0, 0, 7, 0, 9, 14, 0, 0, 0],
                  [0, 0, 0, 9, 0, 10, 0, 0, 0],
                  [0, 0, 4, 14, 10, 0, 2, 0, 0],
                  [0, 0, 0, 0, 0, 2, 0, 1, 6],
                  [8, 11, 0, 0, 0, 0, 1, 0, 7],
                  [0, 0, 2, 0, 0, 0, 6, 7, 0]])  # undirected graph
    B = np.array([[0, 1, 0, 4, 4, 0],
                  [0, 0, 0, 2, 0, 0],
                  [0, 0, 0, 0, 0, 1],
                  [0, 0, 2, 0, 3, 0],
                  [0, 0, 0, 0, 0, 3],
                  [0, 0, 0, 0, 0, 0]])  # directed graph
    graph.generate(A)
    mat = graph.getAdjMatrix()
    print(mat)
