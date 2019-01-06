from heapq import nlargest
from random import Random

from neat.genetics import GeneRecord, Individual, Mutation
from neat.speciation import SpeciationModel

random = Random()


class FitnessDictionary(dict):
    def __init__(self, members):
        super().__init__(self)
        for individual in members:
            self[individual] = 0


class Population:
    def __init__(self, individuals, speciation_model, gene_record, species_list=None):
        self.individuals = individuals
        self.speciation_model = speciation_model
        self.gene_record = gene_record
        self.species_list = species_list
        if self.species_list is None:
            self.species_list = self.speciation_model.categorise_individuals(self.individuals)

    def get_subpopulation(self, individuals):
        return Population(individuals, self.speciation_model, self.gene_record, self.species_list)

    @property
    def size(self):
        return len(self.individuals)


class EvolutionService:
    CROSS_CHANCE = 0.75
    EDGE_DISABLE_CHANCE = 0.75
    NEW_NODE_MUTATION_CHANCE = 0.03
    NEW_EDGE_MUTATION_CHANCE = 0.05
    WEIGHTS_MUTATION_CHANCE = 0.8
    WEIGHT_RESET_CHANCE = 0.1
    INTERSPECIES_CROSS_CHANCE = 0.001
    SPECIES_MINIMUM_CARRY_FORWARD_SIZE = 6

    @staticmethod
    def adjust_fitness_for_species(population, fitness_dictionary):
        for species in population.species_list:
            species_size = len(species)
            total = 0
            for individual in species:
                fitness = fitness_dictionary[individual] / species_size
                fitness_dictionary[individual] = fitness
                total += fitness
            fitness_dictionary[species.index] = total
            print(species.index, "(", species_size, ")", ":", fitness_dictionary[species.index],
                  "(", species_size * max(fitness_dictionary[individual] for individual in list(species)), ")")

    @staticmethod
    def get_n_fittest_individuals(individuals, fitness_dictionary, n):
        return set(nlargest(n, individuals, key=lambda individual: fitness_dictionary[individual]))

    @staticmethod
    def generate_initial_population(population_size, input_node_count, output_node_count):
        gene_record = GeneRecord(input_node_count, output_node_count)
        individuals = [Individual(gene_record) for _ in range(population_size)]
        speciation_model = SpeciationModel()
        return Population(individuals, speciation_model, gene_record)

    @staticmethod
    def generate_offspring(population, fitness_dictionary):
        offspring_individuals = []
        total_fitness = 0
        for species in population.species_list:
            total_fitness += fitness_dictionary[species.index]
            if len(species) > EvolutionService.SPECIES_MINIMUM_CARRY_FORWARD_SIZE:
                fittest_member = max(tuple(individual for individual in species), key=lambda x: fitness_dictionary[x])
                offspring_individuals.append(fittest_member)
        vacancies = len(population.individuals) - len(offspring_individuals)

        for species in population.species_list:
            fitness = fitness_dictionary[species.index]
            if fitness == 0 or total_fitness == 0:
                continue
            offspring_count = round(vacancies * (fitness / total_fitness))
            total_fitness -= fitness
            vacancies -= offspring_count

            species_individuals = list(species)
            species_fitnesses = [fitness_dictionary[individual] for individual in species_individuals]

            for _ in range(offspring_count):  # Mutation
                if random.random() < EvolutionService.CROSS_CHANCE:
                    if len(population.species_list) > 1 and random.random() < EvolutionService.INTERSPECIES_CROSS_CHANCE:
                        parent_1 = random.choices(species_individuals, weights=species_fitnesses)[0]
                        other_species_individuals = list(set(population.individuals).difference(species))
                        other_species_fitnesses = [fitness_dictionary[individual] for individual in other_species_individuals]
                        parent_2 = random.choices(other_species_individuals, weights=other_species_fitnesses)[0]
                    else:
                        parent_1, parent_2 = random.choices(species_individuals, weights=species_fitnesses, k=2)
                    offspring = Mutation.cross(population.gene_record, parent_1, parent_2, EvolutionService.EDGE_DISABLE_CHANCE)
                else:
                    parent = random.choices(species_individuals, weights=species_fitnesses, k=1)[0]
                    offspring = parent.clone()
                if random.random() < EvolutionService.NEW_NODE_MUTATION_CHANCE:
                        Mutation.add_node(population.gene_record, offspring)
                if random.random() < EvolutionService.NEW_EDGE_MUTATION_CHANCE:
                        Mutation.add_edge(population.gene_record, offspring)
                if random.random() < EvolutionService.WEIGHTS_MUTATION_CHANCE:
                    Mutation.mutate_weights(offspring, EvolutionService.WEIGHT_RESET_CHANCE)
                offspring_individuals.append(offspring)

        return offspring_individuals


class EvolutionModel:
    def __init__(self, fitness_function, population_size, input_node_count, output_node_count):
        self.fitness_function = fitness_function
        self.population = EvolutionService.generate_initial_population(population_size, input_node_count,
                                                                       output_node_count)
        self.fitness_dictionary = None
        self.fittest_individual_this_generation = None
        self.fittest_individual_overall = None
        self.evaluate_fitness_for_population()
        EvolutionService.adjust_fitness_for_species(self.population, self.fitness_dictionary)

    def evolve_one_generation(self):
        offspring_individuals = EvolutionService.generate_offspring(self.population, self.fitness_dictionary)
        self.population = Population(offspring_individuals, self.population.speciation_model,
                                     self.population.gene_record)
        self.population.speciation_model.update_holotypes(self.population.species_list)
        self.evaluate_fitness_for_population()
        EvolutionService.adjust_fitness_for_species(self.population, self.fitness_dictionary)

    def evaluate_fitness_for_population(self):
        self.fittest_individual_this_generation = None
        self.fitness_dictionary = FitnessDictionary(self.population.individuals)

        scores = self.fitness_function(self.population)

        for i in range(self.population.size):
            individual = self.population.individuals[i]
            score = scores[i]
            self.fitness_dictionary[individual] = score
            if self.fittest_individual_this_generation is None or score > self.fittest_individual_this_generation[1]:
                self.fittest_individual_this_generation = (individual, score)

        if self.fittest_individual_overall is None or self.fittest_individual_this_generation[1] > self.fittest_individual_overall[1]:
            self.fittest_individual_overall = self.fittest_individual_this_generation

    def get_fittest_individual_this_generation(self):
        return self.fittest_individual_this_generation[0]

    def get_fittest_individual_overall(self):
        return self.fittest_individual_overall[0]

    def get_gene_record(self):
        return self.population.gene_record


class CoevolutionModel:
    REFERENCE_POPULATION_PORTION = 0.2
    REFERENCE_POPULATION_RANDOM_PORTION = 0.5

    def __init__(self, fitness_function, population_size, input_node_count, output_node_count):
        self.fitness_function = fitness_function
        self.population_1 = EvolutionService.generate_initial_population(population_size, input_node_count,
                                                                         output_node_count)
        self.population_2 = EvolutionService.generate_initial_population(population_size, input_node_count,
                                                                         output_node_count)
        self.fitness_dictionary_1 = None
        self.fitness_dictionary_2 = None
        self.fittest_pair_this_generation = None
        self.fittest_pair_overall = None
        self.evaluate_fitness_for_both_populations()

    def evolve_one_generation(self):
        offspring_individuals_1 = EvolutionService.generate_offspring(self.population_1, self.fitness_dictionary_1)
        self.population_1.speciation_model.update_holotypes(self.population_1.species_list)
        self.population_1 = Population(offspring_individuals_1, self.population_1.speciation_model,
                                       self.population_1.gene_record)
        reference_population_2 = self.generate_reference_population(self.population_2, self.fitness_dictionary_2)
        self.evaluate_fitness_for_population_1(reference_population_2)
        EvolutionService.adjust_fitness_for_species(self.population_1, self.fitness_dictionary_1)
        print("----------------------------")

        offspring_individuals_2 = EvolutionService.generate_offspring(self.population_2, self.fitness_dictionary_2)
        self.population_2.speciation_model.update_holotypes(self.population_2.species_list)
        self.population_2 = Population(offspring_individuals_2, self.population_2.speciation_model,
                                       self.population_2.gene_record)
        reference_population_1 = self.generate_reference_population(self.population_1, self.fitness_dictionary_1)
        self.evaluate_fitness_for_population_2(reference_population_1)
        EvolutionService.adjust_fitness_for_species(self.population_2, self.fitness_dictionary_2)

    def evaluate_fitness_for_population_1(self, reference_population_2):
        self.fittest_pair_this_generation = None
        scores = self.fitness_function(self.population_1, reference_population_2)
        self.fitness_dictionary_1 = FitnessDictionary(self.population_1.individuals)

        for i in range(self.population_1.size):
            individual_1 = self.population_1.individuals[i]
            scores_for_individual_1 = scores[i]
            score = max(scores_for_individual_1)
            self.fitness_dictionary_1[individual_1] = score
            if self.fittest_pair_this_generation is None or score > self.fittest_pair_this_generation[1]:
                j = scores_for_individual_1.index(score)
                individual_2 = reference_population_2.individuals[j]
                self.fittest_pair_this_generation = ((individual_1, individual_2), score)

        if self.fittest_pair_overall is None or self.fittest_pair_this_generation[1] > self.fittest_pair_overall[1]:
            self.fittest_pair_overall = self.fittest_pair_this_generation

    def evaluate_fitness_for_population_2(self, reference_population_1):
        self.fittest_pair_this_generation = None
        scores = self.fitness_function(reference_population_1, self.population_2)
        self.fitness_dictionary_2 = FitnessDictionary(self.population_2.individuals)

        for j in range(self.population_2.size):
            individual_2 = self.population_2.individuals[j]
            scores_for_individual_2 = [scores[i][j] for i in range(reference_population_1.size)]
            score = max(scores_for_individual_2)
            self.fitness_dictionary_2[individual_2] = score
            if self.fittest_pair_this_generation is None or score > self.fittest_pair_this_generation[1]:
                i = scores_for_individual_2.index(score)
                individual_1 = reference_population_1.individuals[i]
                self.fittest_pair_this_generation = ((individual_1, individual_2), score)

        if self.fittest_pair_overall is None or self.fittest_pair_this_generation[1] > self.fittest_pair_overall[1]:
            self.fittest_pair_overall = self.fittest_pair_this_generation

    def evaluate_fitness_for_both_populations(self):
        self.fitness_dictionary_1 = FitnessDictionary(self.population_1.individuals)
        self.fitness_dictionary_2 = FitnessDictionary(self.population_2.individuals)

        scores = self.fitness_function(self.population_1, self.population_2)

        for i in range(self.population_1.size):
            individual_1 = self.population_1.individuals[i]
            for j in range(self.population_2.size):
                individual_2 = self.population_2.individuals[j]

                pair_fitness = scores[i][j]
                self.fitness_dictionary_1[individual_1] = max(self.fitness_dictionary_1[individual_1], pair_fitness)
                self.fitness_dictionary_2[individual_2] = max(self.fitness_dictionary_2[individual_2], pair_fitness)

        EvolutionService.adjust_fitness_for_species(self.population_1, self.fitness_dictionary_1)
        print("----------------------------")
        EvolutionService.adjust_fitness_for_species(self.population_2, self.fitness_dictionary_2)

    @staticmethod
    def generate_reference_population(population, fitness_dictionary):
        total_size = max(1, round(CoevolutionModel.REFERENCE_POPULATION_PORTION * population.size))
        n_random = int(total_size * CoevolutionModel.REFERENCE_POPULATION_RANDOM_PORTION)
        n_non_random = total_size - n_random

        individuals = EvolutionService.get_n_fittest_individuals(population.individuals, fitness_dictionary, n_non_random)
        random_pool = [individual for individual in population.individuals if individual not in individuals]
        individuals.update(random.choices(random_pool, k=n_random))

        return population.get_subpopulation(list(individuals))

    def get_fittest_pair_this_generation(self):
        return self.fittest_pair_this_generation[0]

    def get_fittest_pair_overall(self):
        return self.fittest_pair_overall[0]

    def get_gene_records(self):
        return self.population_1.gene_record, self.population_2.gene_record
