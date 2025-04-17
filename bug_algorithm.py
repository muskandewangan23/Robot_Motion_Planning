import numpy as np
import matplotlib.pyplot as plt
grid_size = (20, 20)
start = (2, 2)
goal = (18, 18)
obstacles = {(10, y) for y in range(5, 16)}

def is_free(cell):
    return cell not in obstacles and 0 <= cell[0] < grid_size[0] and 0 <= cell[1] < grid_size[1]

def bug1(start, goal):
    path = [start]
    current = start
    while current != goal:
        next_move = (current[0] + np.sign(goal[0] - current[0]),
                     current[1] + np.sign(goal[1] - current[1]))
        if is_free(next_move):
            current = next_move
        else:
            current = follow_boundary(current, goal)
        path.append(current)
    return path

def follow_boundary(start, goal):
    boundary = []
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    visited = set()
    current = start
    first_hit = start
    best_exit = None
    min_dist = float('inf')
    for dx, dy in directions:
        neighbor = (current[0] + dx, current[1] + dy)
        if not is_free(neighbor):
            current = neighbor
            break
    initial_position = current
    for _ in range(500):
        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if neighbor not in visited and not is_free(neighbor):
                visited.add(current)
                boundary.append(current)
                current = neighbor
                break
        dist_to_goal = abs(current[0] - goal[0]) + abs(current[1] - goal[1])
        if dist_to_goal < min_dist:
            min_dist = dist_to_goal
            best_exit = current
        if current == initial_position and len(boundary) > 1:
            return best_exit if best_exit else start
    return best_exit if best_exit else start

path = bug1(start, goal)

def plot_grid():
    plt.figure(figsize=(6, 6))
    plt.xlim(0, grid_size[0])
    plt.ylim(0, grid_size[1])
    for obs in obstacles:
        plt.scatter(obs[0], obs[1], c='black', marker='s')
    path_x, path_y = zip(*path)
    plt.plot(path_x, path_y, marker='o', linestyle='-')
    plt.scatter(*start, color='green', label='Start')
    plt.scatter(*goal, color='red', label='Goal')
    plt.legend()
    plt.show()
plot_grid()
