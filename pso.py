import random
from random import sample
import numpy as np
import cv2
import math

W = 0.5
c1 = 0.8
c2 = 0.9

n_iterations = int(input("Inform the number of iterations: "))
target_error = float(input("Inform the target error: "))
n_particles = int(input("Inform the number of particles: "))


class Particle:
    def __init__(self):
        limits = flow.shape
        self.position = np.array(
            [random.randint(0, limits[0]), random.randint(0, limits[1])]
        )
        self.pbest_position = self.position
        self.pbest_value = float("inf")
        self.velocity = np.array([0, 0])

    def __str__(self):

        print(self.position, " meu pbest is ", self.pbest_position)

    def move(self):
        self.position = self.position + self.velocity


class Space:
    def __init__(self, target, target_error, n_particles):
        self.target = target
        self.target_error = target_error
        self.n_particles = n_particles
        self.particles = []
        self.gbest_value = float("inf")
        self.gbest_position = np.array([random.random() * 50, random.random() * 50])

    def print_particles(self):
        for particle in self.particles:
            particle.__str__()

    def fitness(self, particle):
        # return particle.position[0] ** 2 + particle.position[1] ** 2 + 1
        return sum(
            flow[math.floor(particle.position[0]), math.floor(particle.position[1])]
        )

    def set_pbest(self):
        for particle in self.particles:
            fitness_cadidate = self.fitness(particle)
            if particle.pbest_value > fitness_cadidate:
                particle.pbest_value = fitness_cadidate
                particle.pbest_position = particle.position

    def set_gbest(self):
        for particle in self.particles:
            best_fitness_candidate = self.fitness(particle)
            if self.gbest_value > best_fitness_candidate:
                self.gbest_value = best_fitness_candidate
                self.gbest_position = particle.position

    def move_particles(self):
        for particle in self.particles:
            global W
            new_velocity = (
                (W * particle.velocity)
                + (c1 * random.random()) * (particle.pbest_position - particle.position)
                + (random.random() * c2) * (self.gbest_position - particle.position)
            )
            particle.velocity = new_velocity
            particle.move()


frame1 = cv2.imread("/home/michael/Documents/AIROB/Robot-vision/data_1/0000000000.png")
frame2 = cv2.imread("/home/michael/Documents/AIROB/Robot-vision/data_1/0000000001.png")

image1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

hsv = np.zeros_like(frame1)
hsv[..., 1] = 255
flow = None

flow = cv2.calcOpticalFlowFarneback(
    image1,
    image2,
    None,  # flow
    0.5,  # pyr_scale
    3,  # levels
    np.random.randint(3, 20),  # winsize
    3,  # iterations
    5,  # poly_n
    1.2,  # poly_sigma
    0,  # flags
)


search_space = Space(0, target_error, n_particles)
particles_vector = [Particle() for _ in range(search_space.n_particles)]
search_space.particles = particles_vector
search_space.print_particles()

iteration = 0
convergence = []
while iteration < n_iterations:
    search_space.set_pbest()
    search_space.set_gbest()
    convergence.append(search_space.gbest_value)
    if abs(search_space.gbest_value - search_space.target) <= search_space.target_error:
        break

    search_space.move_particles()
    iteration += 1

print(
    "The best solution is: ",
    search_space.gbest_position,
    " in n_iterations: ",
    iteration,
)

