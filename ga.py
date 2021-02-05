import random


class GA_searcher():

    def __init__(self, num_in_generation, iteration, stop_criterion, mut_rate, cross_rate):
        self.generation = [Individual()]*num_in_generation
        self.fitness = [0]*num_in_generation
        self.top_n = top_n
        self.stop_criterion = stop_criterion
        self.iteration = iteration
        self.mut_rate = mut_rate
        self.cross_rate = cross_rate

    def get_generation(self):
        return self.generation

    def get_fitness(self):
        return self.eval()

    def eval(self, top_n):
        res = []
        for individual in self.get_generation():
            ind_fitness = individual.eval_alone()
            res.append((individual, ind_fitness))
            # in descending order
            res = sorted(res, lambda x: x[1], reverse=True)
        fittest_val, fittest = zip(*res[:top_n])
        return fittest_val, fittest

    def is_stop_criterion(self, top_n, stop_criterion):
        fittest_val, fittest = self.eval(top_n)
        if fittest_val > stop_criterion:
            return True

    def set_initial_population(self):
        random_ind = Operator().get_random()
        random * num_in_generation

    def evolve(self):
        initial_population =
        while self.is_stop_criterion(self.top_n, self.stop_criterion) or iteration < self.iteration


class Individual():
    def __init__(self):
        self.fitness = 0.0

    def eval_alone(self):


class Operator():
    def __init__(self, operator_where_how):
        self.

    def get_random(self, amount=1):
