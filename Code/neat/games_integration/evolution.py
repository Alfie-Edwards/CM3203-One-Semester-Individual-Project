from neat.evolution import EvolutionModel, CoevolutionModel


def create_evolution_model(fitness_function, population_size, state_size, n_actions):
    return EvolutionModel(fitness_function, population_size, state_size + 1, n_actions)


def create_coevolution_model(fitness_function, population_size, state_size, n_actions, comms_size):
    return CoevolutionModel(fitness_function, population_size, state_size + comms_size + 1, n_actions + comms_size)
