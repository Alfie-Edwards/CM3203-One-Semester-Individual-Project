from random import Random
from copy import deepcopy
import sys

import math

random = Random()


class Individual:

    def __init__(self, gene_record=None):
        self.edge_weights = {}
        self.edges = set()
        self.disabled_edges = set()
        self.incoming_edges_map = {}
        self.node_order = []

        if gene_record:
            for input_node in gene_record.input_nodes:
                for output_node in gene_record.output_nodes:
                    edge = gene_record.register_edge(input_node, output_node)
                    self.add_edge(edge, output_node, 0.1)

    @property
    def nodes(self):
        return set(self.node_order)

    def add_node(self, node, global_node_order):
        i = 0
        for global_node in global_node_order:
            if i == len(self.node_order):
                break
            if global_node == self.node_order[i]:
                if self.node_order[i] == node:
                    return
                i += 1
            elif global_node == node:
                break
        self.node_order.insert(i, node)

    def add_edge(self, edge, end_node, weight):
        self.edge_weights[edge] = weight
        self.edges.add(edge)
        if end_node not in self.incoming_edges_map:
            self.incoming_edges_map[end_node] = set()
        self.incoming_edges_map[end_node].add(edge)

    def disable_edge(self, edge, end_node):
        self.disabled_edges.add(edge)
        self.incoming_edges_map[end_node].remove(edge)

    @property
    def enabled_edges(self):
        return self.edges.difference(self.disabled_edges)

    def clone(self):
        new_individual = Individual()
        new_individual.edge_weights = deepcopy(self.edge_weights)
        new_individual.edges = deepcopy(self.edges)
        new_individual.disabled_edges = deepcopy(self.disabled_edges)
        new_individual.node_order = deepcopy(self.node_order)
        new_individual.incoming_edges_map = deepcopy(self.incoming_edges_map)
        return new_individual


class GeneRecord:

    def __init__(self, input_node_count, output_node_count):
        self.input_node_count = input_node_count
        self.output_node_count = output_node_count
        self.edge_definitions = []  # edge : (start_node, end_node)
        self.node_definitions = []  # node : edge
        self.incoming_edges_map = {}
        self.input_nodes = list(range(-1, -input_node_count - 1, -1))
        self.output_nodes = list(range(sys.maxsize - output_node_count + 1, sys.maxsize + 1))
        self.node_order = []

    def register_edge(self, start_node, end_node):
        edge_definition = (start_node, end_node)
        if edge_definition in self.edge_definitions:
            return self.edge_definitions.index(edge_definition)

        edge = len(self.edge_definitions)
        self.edge_definitions.append(edge_definition)
        if end_node in self.incoming_edges_map:
            self.incoming_edges_map[end_node].add(edge)
        else:
            self.incoming_edges_map[end_node] = {edge}

        return edge

    def register_node(self, edge):
        if edge in self.node_definitions:
            return self.node_definitions.index(edge)

        node = len(self.node_definitions)
        self.node_definitions.append(edge)
        start_node, end_node = self.edge_definitions[edge]

        if start_node in self.intermediate_nodes:
            start_node_index = self.node_order.index(start_node)
        elif start_node in self.input_nodes:
            start_node_index = -1
        else:
            raise Exception("Invalid start node", start_node, "found for edge", edge)

        if end_node in self.intermediate_nodes:
            end_node_index = self.node_order.index(end_node)
        elif end_node in self.output_nodes:
            end_node_index = len(self.node_definitions)
        else:
            raise Exception("Invalid end node", end_node, "found for edge", edge)

        new_node_index = int(math.ceil((start_node_index + end_node_index) / 2))
        self.node_order.insert(new_node_index, node)

        return node

    def get_incoming_edges(self, node):
        return self.incoming_edges_map[node]

    @property
    def edges(self):
        return set(range(len(self.edge_definitions)))

    @property
    def intermediate_nodes(self):
        return set(range(len(self.node_definitions)))


class Mutation:
    CROSS_EDGE_DISABLE_CHANCE = 0.75
    WEIGHT_MUTATION_CHANCE = 0.9

    @staticmethod
    def add_edge(gene_record, individual):
        start_nodes = gene_record.input_nodes + individual.node_order
        end_nodes = individual.node_order + gene_record.output_nodes
        start_node_indices = list(range(len(start_nodes)))
        input_nodes_removed = 0

        while start_node_indices:
            start_node_index = random.choice(start_node_indices)
            n_input_nodes = len(gene_record.input_nodes) - input_nodes_removed
            end_node_index_offset = max(0, start_node_index - n_input_nodes + 1)
            end_nodes_indices = list(range(end_node_index_offset, len(end_nodes)))
            while end_nodes_indices:
                end_node_index = random.choice(end_nodes_indices)
                start_node = start_nodes[start_node_index]
                end_node = end_nodes[end_node_index]
                edge = gene_record.register_edge(start_node, end_node)

                if edge not in individual.edges:
                    weight = random.normalvariate(0, 0.5)
                    individual.add_edge(edge, end_node, weight)
                    return

                end_nodes_indices.remove(end_node_index)
            start_node_indices.remove(start_node_index)
            if end_node_index_offset == 0:
                input_nodes_removed += 1

    @staticmethod
    def add_node(gene_record, individual):
        if not individual.edges or not individual.enabled_edges:
            return
        edge = random.choice(tuple(individual.enabled_edges))
        start_node, end_node = gene_record.edge_definitions[edge]
        individual.disable_edge(edge, end_node)

        new_node = gene_record.register_node(edge)
        individual.add_node(new_node, gene_record.node_order)

        in_edge = gene_record.register_edge(start_node, new_node)
        out_edge = gene_record.register_edge(new_node, end_node)
        in_weight = random.normalvariate(0, 0.5)
        out_weight = random.normalvariate(0, 0.5)
        individual.add_edge(in_edge, new_node, in_weight)
        individual.add_edge(out_edge, end_node, out_weight)
        a=0

    @staticmethod
    def cross(gene_record, primary_individual, secondary_individual, edge_disable_chance):
        new_individual = primary_individual.clone()
        for edge in new_individual.edges:
            if random.random() < 0.5 and edge in secondary_individual.edges:
                new_individual.edge_weights[edge] = secondary_individual.edge_weights[edge]
            if edge in primary_individual.disabled_edges:
                if random.random() >= edge_disable_chance:
                    new_individual.disabled_edges.remove(edge)
                    new_individual.incoming_edges_map[gene_record.edge_definitions[edge][1]].add(edge)
            elif edge in primary_individual.disabled_edges and random.random() < edge_disable_chance:
                new_individual.disable_edge(edge, gene_record.edge_definitions[edge][1])
        return new_individual

    @staticmethod
    def mutate_weights(individual, reset_chance):
        for edge in individual.enabled_edges:
            if random.random() < reset_chance:
                individual.edge_weights[edge] = random.normalvariate(0, 0.5)
            else:
                individual.edge_weights[edge] += random.normalvariate(0, 0.5)

