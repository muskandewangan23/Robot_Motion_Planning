import numpy as np
import matplotlib.pyplot as plt
grid_size = (20, 20)
start = (2, 2)
goal = (18, 18)
obstacles = {(10, y) for y in range(5, 16)}

def potential_field(grid_size, goal, obstacles):
    field = np.zeros(grid_size)
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            if (x, y) in obstacles:
                field[x, y] = np.inf
            else:
                field[x, y] = np.linalg.norm(np.array([x, y]) - np.array(goal))**2
    return field

field = potential_field(grid_size, goal, obstacles)

def gradient_descent_path(start, goal, field, obstacles):
    path = [start]
    current = start
    while current != goal:
        x, y = current
        min_potential = field[x, y]
        next_move = current
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor = (x + dx, y + dy)
                if 0 <= neighbor[0] < grid_size[0] and 0 <= neighbor[1] < grid_size[1]:
                    if neighbor not in obstacles and field[neighbor] < min_potential:
                        min_potential = field[neighbor]
                        next_move = neighbor
        if next_move == current:
            break
        current = next_move
        path.append(current)
    return path

path = gradient_descent_path(start, goal, field, obstacles)
plt.figure(figsize=(6, 6))
plt.imshow(field.T, origin='lower', cmap='jet', interpolation='nearest')
plt.colorbar(label='Potential')
for obs in obstacles:
    plt.scatter(obs[0], obs[1], c='black', marker='s')
path_x, path_y = zip(*path)
plt.plot(path_x, path_y, marker='o', linestyle='-', color='white')
plt.scatter(*start, color='green', label='Start')
plt.scatter(*goal, color='red', label='Goal')
plt.legend()
plt.title("Potential Field with Path")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()