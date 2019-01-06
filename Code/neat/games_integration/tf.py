import tensorflow as tf
import numpy as np


class Graph:

    tf_session = None

    def __init__(self, gene_record):
        self.input_nodes = tf.placeholder(tf.float32, [None, None, len(gene_record.input_nodes)], name="inputs")
        self.__node_dictionary = {}  # node : node_placeholder
        self.__edge_dictionary = {}  # end_node : parent_nodes_variable, incoming_edges, weights_placeholder
        self.gene_record = gene_record
        self.latest_node = len(self.gene_record.node_definitions) - 1
        self.latest_edge = len(self.gene_record.edge_definitions) - 1
        self.variables = []
        self.initialisation_operations = {}
        self.initialiser = None

        input_nodes_variable = tf.Variable(self.input_nodes, trainable=False, validate_shape=False)
        self.variables.append(input_nodes_variable)
        individual_input_nodes = tf.split(input_nodes_variable, len(gene_record.input_nodes), axis=2)
        input_node_order = sorted(gene_record.input_nodes)
        for i in range(len(gene_record.input_nodes)):
            self.__node_dictionary[input_node_order[i]] = individual_input_nodes[i]

        node_list = gene_record.node_order + sorted(gene_record.output_nodes)
        for node in node_list:
            self.__add_node(node)
        self.output_nodes = tf.concat([self.__node_dictionary[output_node]
                                       for output_node in sorted(self.gene_record.output_nodes)], axis=2)
        self.initialiser = tf.variables_initializer(self.variables)

    def update_structure(self):
        new_latest_node = len(self.gene_record.node_definitions) - 1
        new_latest_edge = len(self.gene_record.edge_definitions) - 1
        new_nodes = range(self.latest_node + 1, new_latest_node + 1)
        new_edges = list(range(self.latest_edge + 1, new_latest_edge + 1))

        for node in new_nodes:
            self.__add_node(node)

        nodes_to_refresh = set(self.gene_record.edge_definitions[edge][1] for edge in new_edges)
        nodes_to_refresh.difference_update(new_nodes)
        for node in nodes_to_refresh:
            self.__update_node(node)
        self.latest_node = new_latest_node
        self.latest_edge = new_latest_edge
        self.initialiser = tf.variables_initializer(self.variables)

    def __add_node(self, node):
        incoming_edges = sorted(self.gene_record.incoming_edges_map[node])
        parent_nodes = [self.gene_record.edge_definitions[edge][0] for edge in incoming_edges]
        parent_tensor = tf.concat([self.__node_dictionary[node] for node in parent_nodes], axis=2)
        parent_nodes_variable = tf.Variable(initial_value=0, dtype=tf.float32, trainable=False, validate_shape=False)
        self.initialisation_operations[node] = tf.assign(parent_nodes_variable, parent_tensor, validate_shape=False)
        weights_placeholder = tf.placeholder(tf.float32, shape=[None, None, 1], name=str(node)+"_weights")
        weights_variable = tf.Variable(weights_placeholder, trainable=False, validate_shape=False)
        self.variables.append(parent_nodes_variable)
        self.variables.append(weights_variable)
        self.__node_dictionary[node] = tf.nn.relu(tf.matmul(parent_nodes_variable, weights_variable))
        self.__edge_dictionary[node] = (parent_nodes_variable, incoming_edges, weights_placeholder)

    def __update_node(self, node):
        parent_nodes_variable, _, weights_placeholder = self.__edge_dictionary[node]
        incoming_edges = sorted(self.gene_record.incoming_edges_map[node])
        parent_nodes = [self.gene_record.edge_definitions[edge][0] for edge in incoming_edges]
        parent_tensor = tf.concat([self.__node_dictionary[node] for node in parent_nodes], axis=2)
        self.initialisation_operations[node] = tf.assign(parent_nodes_variable, parent_tensor, validate_shape=False)
        self.__edge_dictionary[node] = (parent_nodes_variable, incoming_edges, weights_placeholder)

    def feed_forward(self, feed_dict):
        self.tf_session.run(self.initialiser, feed_dict=feed_dict)

        if self.tf_session is None:
            raise ValueError("Must call start_session() before calling feed_forward()")
        for node in self.gene_record.node_order + self.gene_record.output_nodes:
            if node in self.initialisation_operations:
                self.tf_session.run(self.initialisation_operations[node])

        return self.tf_session.run(self.output_nodes)

    def generate_feed_dict(self, individuals, input_values_buffer):
        feed_dict = {self.input_nodes: input_values_buffer}

        for end_node in self.__edge_dictionary:
            _, incoming_edges, weights_placeholder = self.__edge_dictionary[end_node]
            feed_dict[weights_placeholder] = [[(individual.edge_weights[edge] if edge in individual.edges and edge not in individual.disabled_edges else 0,) for edge in incoming_edges] for individual in individuals]
        return feed_dict

    @staticmethod
    def start_session():
        if Graph.tf_session is not None:
            return
        Graph.tf_session = tf.Session()

    @staticmethod
    def end_session():
        if Graph.tf_session is None:
            return
        Graph.tf_session.close()
        Graph.tf_session = None
