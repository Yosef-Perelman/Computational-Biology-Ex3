import random
import json
import numpy as np


class GeneticAlgorithm:
    def __init__(self, population_size=500, replication_rate=0.05,
                 mutation_rate=0.3, max_generations=500):
        self.population_size = population_size
        self.replication_rate = replication_rate
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.generation = 0
        self.population = []
        self.fitness = []
        self.weights = []
        self.stop = False

    def initialize_population(self):
        self.population = [self.generate_individual()
                           for i in range(self.population_size)]

    def evaluate_population_fitness(self):
        self.fitness = [self.eval_fn(individual)
                        for individual in self.population]

    def update_ranks(self):
        sorted_fitness = sorted(self.fitness, reverse=True)
        self.ranks = [sorted_fitness.index(fitness) + 1 for fitness in self.fitness]

    def selection_individual(self):
        # Select an individual using roulette wheel selection
        weights = self.ranks
        index = random.choices(range(self.population_size),
                               weights=weights, k=1)[0]
        return index

    def run(self):
        self.initialize_population()
        self.evaluate_population_fitness()
        self.update_ranks()

        # Keep track of the best individual and its fitness
        best_fitness = min(self.fitness)
        worst_fitness = max(self.fitness)
        avg_fitness = sum(self.fitness) / self.population_size
        best_individual = self.population[self.fitness.index(
            best_fitness)].copy()
        print(f'gen {self.generation}: best = {best_fitness} | worst = {worst_fitness} | avg = {avg_fitness} | '
              f'success rates = {((15000 - best_fitness) / 15000) * 100}%')

        # Loop through the generations until the termination criteria is met
        while self.generation < self.max_generations and self.stop == False:
            new_population = []

            # select individual for replication
            replication_size = int(self.replication_rate * self.population_size)
            while len(new_population) < replication_size:
                new_population.append(best_individual.copy())

            # select individuals for crossover
            selected = []
            while len(new_population) < self.population_size:
                parent1, parent2 = self.selection_individual(), self.selection_individual()
                while parent1 == parent2 or (parent1, parent2) in selected:
                    parent1, parent2 = self.selection_individual(), self.selection_individual()
                selected.append((parent1, parent2))
                offspring = self.crossover(self.population[parent1], self.population[parent2])
                new_population.append(offspring)

            # Mutate a portion of the population
            mutation_size = int(self.population_size * self.mutation_rate)
            mutation_candidates = new_population[1:]
            random.shuffle(mutation_candidates)
            mutation_individuals = random.sample(
                mutation_candidates, mutation_size)
            for individual in mutation_individuals:
                self.mutate(individual)

            # update population
            random.shuffle(new_population)
            self.population = new_population
            self.evaluate_population_fitness()
            self.update_ranks()

            # Update the best individual and its fitness
            new_best_fitness = min(self.fitness)
            new_worst_fitness = max(self.fitness)
            new_best_individual = self.population[self.fitness.index(
                new_best_fitness)].copy()
            if new_best_fitness < best_fitness:
                best_fitness = new_best_fitness
                best_individual = new_best_individual
            avg_fitness = sum(self.fitness) / self.population_size

            # Increment the generation counter
            self.generation += 1

            print(f'gen {self.generation}: best = {best_fitness} | worst = {new_worst_fitness} | avg = {avg_fitness} | '
                  f'success rates = {((15000 - best_fitness) / 15000) * 100}%')

            if ((15000 - best_fitness) / 15000) * 100 > 97.5:
                self.stop = True

        # Return the best individual and its fitness
        return best_individual, best_fitness

    def eval_fn(self, individual):
        raise NotImplementedError()

    def crossover(self, individual1, individual2):
        raise NotImplementedError()

    def mutate(self, individual):
        raise NotImplementedError()

    def generate_individual(self):
        raise NotImplementedError()


class BuildNet(GeneticAlgorithm):
    def __init__(self, data, population_size=100, replication_rate=0.05, mutation_rate=0.3,
                 max_generations=500, first_layer_size=16, middle_layer_size=10, last_layer_size=1):
        super().__init__(population_size=population_size, replication_rate=replication_rate,
                         mutation_rate=mutation_rate, max_generations=max_generations)
        self.data = data
        self.first_layer_size = first_layer_size + 1
        self.middle_layer_size = middle_layer_size + 1
        self.last_layer_size = last_layer_size

    def generate_individual(self):
        middle_weights_values = np.random.uniform(-1, 1, (self.first_layer_size, self.middle_layer_size))
        last_weights_values = np.random.uniform(-1, 1, (self.middle_layer_size, self.last_layer_size))
        individual = [middle_weights_values, last_weights_values]
        return individual

    def eval_fn(self, individual):

        false_predict = 0
        for sample in self.data:
            # Add bias
            input_layer = np.concatenate((sample[0], np.ones(1)))

            middle_layer = np.dot(input_layer, individual[0])
            middle_layer_after_activation = np.maximum(middle_layer, 0)

            last_layer = np.dot(middle_layer_after_activation, individual[1])
            last_layer_after_activation = (1 / (1 + np.exp(-last_layer)))

            prediction = int(last_layer_after_activation[0] >= 0.5)
            if prediction != int(sample[1]):
                false_predict += 1
        return false_predict

    def crossover(self, individual1, individual2):

        random_point1 = random.randint(1, self.first_layer_size * self.middle_layer_size)
        random_point2 = random.randint(1, self.middle_layer_size * self.last_layer_size)

        parent1 = [individual1[0].flatten(), individual1[1].flatten()]
        parent2 = [individual2[0].flatten(), individual2[1].flatten()]

        offspring = [np.concatenate([parent1[0][:random_point1], parent2[0][random_point1:]]),
                     np.concatenate([parent1[1][:random_point2], parent2[1][random_point2:]])]
        offspring = [offspring[0].reshape(individual1[0].shape), offspring[1].reshape(individual1[1].shape)]

        return offspring

    def mutate(self, individual):
        for weights in individual:
            weights += np.random.uniform(-0.1, 0.1, size=weights.shape)


train_set = []
test_set = []
with open('nn0.txt', 'r') as file:
    for line in file:
        sample, label = line.strip().split()
        sample = np.array([int(bit) for bit in sample])

        if len(train_set) < 15000:
            train_set.append([sample, label])
        else:
            test_set.append([sample, label])

genetic_learning = BuildNet(data=train_set)
solution, fitness = genetic_learning.run()

print(f"train set precision: {((15000 - fitness) / 15000) * 100}")

false_predict = 0
for sample in test_set:
    # Add bias
    input_layer = np.concatenate((sample[0], np.ones(1)))

    middle_layer = np.dot(input_layer, solution[0])
    middle_layer_after_activation = np.maximum(middle_layer, 0)

    last_layer = np.dot(middle_layer_after_activation, solution[1])
    last_layer_after_activation = (1 / (1 + np.exp(-last_layer)))

    prediction = int(last_layer_after_activation[0] >= 0.5)
    if prediction != int(sample[1]):
        false_predict += 1

print(f"test set precision: {((5000 - false_predict) / 5000) * 100}")

network_structure = {'first_layer_size': 16,'middle_layer_size': 10,'last_layer_size': 1,}

weights = {'weights1': solution[0].tolist(),'weights2': solution[1].tolist(),}

output = {'network_structure': network_structure,'weights': weights,}

with open('wnet0.json', 'w') as f:
    json.dump(output, f)