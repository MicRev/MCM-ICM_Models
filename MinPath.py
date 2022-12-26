from graph import Graph
import numpy as np


def dijkstra(graph: Graph, start: int):  # must use BFS instead of DFS otherwise you will be trapped in a branch path
    """由图中某一个点开始找到遍历图的最短路径.

    Parameters
    ----------
    graph : Graph
        图对象
    start : int
        起始点的序号

    Returns
    -------
    Graph
        新的图对象, 在原有的图的基础上只保留最短路径而删除其他所有连接.
    """
    dist = [np.inf] * graph.size
    dist[start] = 0
    queue = [graph.nodes[start]]
    while queue:
        node = queue.pop(0)
        pointer = node.index
        for neighb in node.neighbours:
            idx = neighb.index
            # update the shortest path
            if dist[idx] > node.getWeight(neighb) + dist[pointer] > dist[pointer]:
                dist[idx] = node.getWeight(neighb) + dist[pointer]
                neighb.previous = node
                queue.append(neighb)
                # get the next node
        # make the head of the queue remain min
        if len(queue) > 1:
            j = 0
            mindist = dist[pointer]
            for i, n in enumerate(queue[1:]):
                if mindist > dist[n.index]:
                    mindist = dist[n.index]
                    j = i
            queue[0], queue[j] = queue[j], queue[0]

    new_graph = Graph(graph.size)
    for node in graph.nodes:
        path = [node.index]
        a = '''节点{}，最短路径 '''.format(node.index)
        cur = node
        distance = 0
        while cur.previous:
            if new_graph.nodes[cur.index] not in new_graph.nodes[cur.previous.index].neighbours:
                new_graph.nodes[cur.previous.index].addNeighbour(new_graph.nodes[cur.index],
                                                                  cur.previous.get_weight(cur))
            path.insert(0, cur.previous.index)
            distance += cur.previous.get_weight(cur)
            cur = cur.previous
        b = ''
        for d in path:
            b += ' -> '
            b += str(d)
        b = b[4:]
        c = '，最短距离为{}'.format(distance)
        print(a + b + c)
    return new_graph


if __name__ == '__main__':
    graph = Graph(6)
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
    graph.generate(B)
    shortest = dijkstra(graph, 0)
    print(shortest.getAdjMatrix())
