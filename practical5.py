import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
map_size = (20, 20)
num_samples = 100
k_nearest = 5
step_size = 1.5
obstacles = [((8, 5), (10, 15))]
start = (2, 2)
goal = (18, 18)

def is_collision_free(p1, p2, obstacles):
    """Checks if a straight-line path between two points collides with any obstacles."""
    x1, y1 = p1
    x2, y2 = p2
    for (x_min, y_min), (x_max, y_max) in obstacles:
        if min(x1, x2) <= x_max and max(x1, x2) >= x_min and min(y1, y2) <= y_max and max(y1, y2) >= y_min:
            return False
    return True

def generate_prm_graph(num_samples, k_nearest, obstacles):
    """Generates a PRM graph with nodes and edges avoiding obstacles."""
    graph = nx.Graph()
    nodes = [start, goal]
    while len(nodes) < num_samples:
        x, y = np.random.uniform(0, map_size[0]), np.random.uniform(0, map_size[1])
        if all(not (obs[0][0] <= x <= obs[1][0] and obs[0][1] <= y <= obs[1][1]) for obs in obstacles):
            nodes.append((x, y))
    for i, node in enumerate(nodes):
        distances = [(np.linalg.norm(np.array(node) - np.array(other)), other) for other in nodes if other != node]
        distances.sort()
        for _, neighbor in distances[:k_nearest]:
            if is_collision_free(node, neighbor, obstacles):
                graph.add_edge(node, neighbor, weight=np.linalg.norm(np.array(node) - np.array(neighbor)))
    return graph

prm_graph = generate_prm_graph(num_samples, k_nearest, obstacles)

def generate_rrt(start, goal, step_size, obstacles, max_iter=5000):
    """Generates an RRT from start to goal, avoiding obstacles."""
    tree = {start: None}
    for _ in range(max_iter):
        rand_point = goal if random.random() < 0.1 else (random.uniform(0, map_size[0]), random.uniform(0, map_size[1]))
        nearest_node = min(tree.keys(), key=lambda n: np.linalg.norm(np.array(n) - np.array(rand_point)))
        direction = np.array(rand_point) - np.array(nearest_node)
        direction = direction / np.linalg.norm(direction) * step_size
        new_node = tuple(np.array(nearest_node) + direction)
        if is_collision_free(nearest_node, new_node, obstacles):
            tree[new_node] = nearest_node
            if np.linalg.norm(np.array(new_node) - np.array(goal)) < step_size:
                tree[goal] = new_node
                break
    path = []
    node = goal if goal in tree else None
    while node:
        path.append(node)
        node = tree[node]
    return tree, path[::-1]

rrt_tree, rrt_path = generate_rrt(start, goal, step_size, obstacles)
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
ax = axes[0]
ax.set_title("Probabilistic Roadmap (PRM)")
ax.set_xlim(0, map_size[0])
ax.set_ylim(0, map_size[1])

for (x_min, y_min), (x_max, y_max) in obstacles:
    ax.fill([x_min, x_max, x_max, x_min], [y_min, y_min, y_max, y_max], 'k')
for edge in prm_graph.edges:
    ax.plot(*zip(*edge), 'gray', linestyle='--', alpha=0.5)
ax.scatter(*zip(*prm_graph.nodes), c='black', s=20, label="PRM Nodes")
ax.scatter(*start, c='green', s=100, label="Start")
ax.scatter(*goal, c='red', s=100, label="Goal")
try:
    prm_path = nx.astar_path(prm_graph, start, goal)
    ax.plot(*zip(*prm_path), 'b', linewidth=2, label="A* Path")
except nx.NetworkXNoPath:
    prm_path = []
    print("No path found in PRM.")
ax.legend()
ax = axes[1]
ax.set_title("Rapidly-exploring Random Tree (RRT)")
ax.set_xlim(0, map_size[0])
ax.set_ylim(0, map_size[1])

for (x_min, y_min), (x_max, y_max) in obstacles:
    ax.fill([x_min, x_max, x_max, x_min], [y_min, y_min, y_max, y_max], 'k')

for node, parent in rrt_tree.items():
    if parent:
        ax.plot([node[0], parent[0]], [node[1], parent[1]], 'gray', linestyle='--', alpha=0.5)

if rrt_path:
    ax.plot(*zip(*rrt_path), 'b', linewidth=2, label="RRT Path")
ax.scatter(*start, c='green', s=100, label="Start")
ax.scatter(*goal, c='red', s=100, label="Goal")
ax.legend()
plt.show()