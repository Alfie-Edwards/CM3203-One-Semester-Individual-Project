from typing import Iterable, Any
from random import Random

random = Random()


class Species(set):
    def __init__(self, index, *args):
        super().__init__(self, *args)
        self.index = index

    def intersection(self, *s: Iterable[Any]):
        intersection = super().intersection(*s)
        new_species = Species(self.index)
        new_species.update(intersection)
        return new_species


class SpeciationModel:

    SPECIATION_THRESHOLD = 3
    EXCESS_FACTOR = 1
    DISJOINT_FACTOR = 1
    DIFFERENCE_FACTOR = 0.4

    def __init__(self):
        self.species_index_counter = 0
        self.holotypes = {}

    def categorise_individuals(self, individuals):
        species_dictionary = {}
        for individual in individuals:
            matches_existing_species = False
            for species_index in sorted(self.holotypes.keys()):
                holotype = self.holotypes[species_index]
                compatibility = self.__compatibility_distance(individual, holotype)
                if compatibility < self.SPECIATION_THRESHOLD:
                    if species_index in species_dictionary:
                        species = species_dictionary[species_index]
                    else:
                        species = Species(species_index)
                        species_dictionary[species_index] = species
                    species.add(individual)
                    matches_existing_species = True
                    break
            if not matches_existing_species:
                species_index = self.species_index_counter
                species = Species(species_index)
                species.add(individual)
                species_dictionary[species_index] = species
                self.holotypes[species_index] = individual
                self.species_index_counter += 1
        return species_dictionary.values()

    def update_holotypes(self, species_list):
        self.holotypes = {species.index: random.choice(tuple(species)) for species in species_list}

    @staticmethod
    def __compatibility_distance(individual_1, individual_2):
        genome_size = max(len(individual_1.edges), len(individual_2.edges))
        excess_factor = SpeciationModel.EXCESS_FACTOR / genome_size
        disjoint_factor = SpeciationModel.DISJOINT_FACTOR / genome_size
        difference_factor = SpeciationModel.DIFFERENCE_FACTOR

        common = individual_1.edges.intersection(individual_2.edges)
        non_common = individual_1.edges.union(individual_2.edges).difference(common)
        excess_threshold = min(max(individual_1.edges), max(individual_2.edges))
        excess = sum(1 for edge in non_common if edge <= excess_threshold)
        disjoint = len(non_common) - excess

        average_difference = sum(abs(individual_1.edge_weights[edge] - individual_2.edge_weights[edge])
                                 for edge in common) / (len(common) or 1)
        return (excess_factor * excess) + (disjoint_factor * disjoint) + (difference_factor * average_difference)
