import json
import os
from copy import deepcopy

import numpy as np

from .gntm import GNTM

file_path = os.path.dirname(os.path.realpath(__file__))
pool_path = os.path.join(file_path, "pool")


def init_pool(current_pool, num_models: int = None, from_file: bool = False):
    """Either num_models or from file has to be specified"""
    for i in range(num_models):
        model = GNTM()
        current_pool.append(model)
        if from_file is False:
            model.get_model().save_weights(
                os.path.join(pool_path, "model_{}.keras".format(i))
            )
    if from_file:
        path, dirs, files = next(os.walk(pool_path))
        for i, file in enumerate(files):
            current_pool[i].get_model().load_weights(os.path.join(pool_path, file))
    return current_pool


def make_love_not_gntm(parent_1, parent_2):
    """Produce offspring with genes from 2 parents."""
    weight_1 = parent_1.get_model().get_weights()
    weight_2 = parent_2.get_model().get_weights()

    new_weight_1 = deepcopy(weight_1)
    new_weight_2 = deepcopy(weight_2)
    for i in range(len(new_weight_1)):
        if i % 2 == 0:
            gene = np.random.randint(0, len(new_weight_1[i]) - 1)
            gene_2 = np.random.randint(0, len(new_weight_1[i]) - 1)
            if gene > gene_2:
                gene, gene_2 = gene_2, gene

        if gene != gene_2:
            new_weight_1[i][gene:gene_2] = weight_2[i][gene:gene_2]
            new_weight_2[i][gene:gene_2] = weight_1[i][gene:gene_2]

    return new_weight_1, new_weight_2


def mutate_models(self, fitness, current_pool):
    """Orchestrate the mutation and gene crossover and set up the new generation."""
    fitness_array = np.array(fitness)
    current_pool = [
        current_pool[i] for i in np.flip(np.argsort(fitness_array)).tolist()
    ]
    print(np.sort(fitness))
    with open(os.path.join(file_path, "fitness_coin_master.json"), "r") as infile:
        fitness_per_gen = json.load(infile)
    fitness_per_gen.append(int(np.sort(fitness)[-1]))
    print("Generation: ", len(fitness_per_gen) - 1)
    with open(os.path.join(file_path, "fitness_coin_master.json"), "w") as infile:
        json.dump(fitness_per_gen, infile)

    top_share = self.max_models // 5

    top_parents = [model for i, model in enumerate(current_pool) if i < top_share]
    current_pool = top_parents
    current_pool[0].get_model().save_weights("gntm.keras")
    for i, parent in enumerate(top_parents[:top_share]):
        for _ in range(2):
            randint = i
            if i > 1:
                randint = np.random.randint(0, i - 1)
            weights_1, weights_2 = make_love_not_gntm(parent, current_pool[randint])
            child_1 = GNTM()
            child_2 = GNTM()
            child_1.set_weights(weights_1)
            child_2.set_weights(weights_2)
            current_pool.append(child_1.cell_division())
            current_pool.append(child_2.cell_division())

    assert len(current_pool) == self.max_models
    for i in range(0, len(current_pool)):
        current_pool[i].get_model().save_weights(
            os.path.join(pool_path, "model_{}.keras".format(i))
        )

    return current_pool
