import numpy as np
import matplotlib.pyplot as plt
grid_size = (20, 20)
obstacles = {(9, y) for y in range(5, 16)} | {(12, y) for y in range(5, 16)}
robot_width = 2
robot_height = 3

def inflate_obstacles(obstacles, robot_width, robot_height):
    inflated_obstacles = set()
    for (x, y) in obstacles:
        for dx in range(-robot_width//2, robot_width//2 + 1):
            for dy in range(-robot_height//2, robot_height//2 + 1):
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < grid_size[0] and 0 <= new_y < grid_size[1]:
                    inflated_obstacles.add((new_x, new_y))
    return inflated_obstacles

inflated_obstacles = inflate_obstacles(obstacles, robot_width, robot_height)

def plot_config_space():
    plt.figure(figsize=(6, 6))
    plt.xlim(0, grid_size[0])
    plt.ylim(0, grid_size[1])
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Configuration Space (Rectangular Robot)")
    for obs in inflated_obstacles:
        plt.scatter(obs[0], obs[1], c='black', marker='s')
    plt.show()

plot_config_space()
