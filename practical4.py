import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon
start = (2, 2)
goal = (18, 18)
obstacles = [
    [(9, 5), (9, 15), (10, 15), (10, 5)]
]

def is_visible(p1, p2, obstacles):
    line = LineString([p1, p2])
    for obs in obstacles:
        polygon = Polygon(obs)
        if line.crosses(polygon) or line.within(polygon):
            return False
    return True

def construct_visibility_graph(obstacles, start, goal):
    G = nx.Graph()
    nodes = [start, goal]
    for obs in obstacles:
        nodes.extend(obs)
    for node in nodes:
        G.add_node(node)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if is_visible(nodes[i], nodes[j], obstacles):
                G.add_edge(nodes[i], nodes[j], weight=np.linalg.norm(np.array(nodes[i]) - np.array(nodes[j])))
    return G

visibility_graph = construct_visibility_graph(obstacles, start, goal)
print("Nodes in visibility graph:", list(visibility_graph.nodes))
print("Edges in visibility graph:", list(visibility_graph.edges))
try:
    path = nx.astar_path(visibility_graph, start, goal)
    print("Path found:", path)
except nx.NetworkXNoPath:
    print("No path found between start and goal.")
    path = []
plt.figure(figsize=(8, 8))
plt.xlim(0, 20)
plt.ylim(0, 20)
for obs in obstacles:
    polygon = Polygon(obs)
    x, y = polygon.exterior.xy
    plt.fill(x, y, 'black')
for edge in visibility_graph.edges:
    x_values = [edge[0][0], edge[1][0]]
    y_values = [edge[0][1], edge[1][1]]
    plt.plot(x_values, y_values, 'k--', alpha=0.3)

if path:
    for i in range(len(path) - 1):
        x_values = [path[i][0], path[i + 1][0]]
        y_values = [path[i][1], path[i + 1][1]]
        plt.plot(x_values, y_values, 'b-', linewidth=2, label="A* Path" if i == 0 else "")
plt.scatter(start[0], start[1], c='green', s=100, label="Start")
plt.scatter(goal[0], goal[1], c='red', s=100, label="Goal")
plt.legend()
plt.title("Visibility Graph with A* Path")
plt.show()