from abc import ABC, abstractmethod
import numpy as np


class AbstractPlaySession(ABC):
    def __init__(self, game_instance):
        self.game_instance = game_instance
        self.current_turn = 0

    def play_turns(self, n=1):
        for i in range(n):
            if self.game_instance.is_terminated():
                return
            self._play_turn()
            self.current_turn += 1

    @abstractmethod
    def _play_turn(self):
        pass

    def play_until_termination(self):
        while not self.game_instance.is_terminated():
            self.play_turns(1)

    def get_current_state(self):
        return self.game_instance.get_state()

    def get_current_score(self):
        return self.game_instance.get_score()

    def get_current_turn(self):
        return self.current_turn

    def is_terminated(self):
        return self.game_instance.is_terminated()

    @staticmethod
    def _feedforward(individual, input_values, gene_record):
        node_values = {}
        input_nodes = sorted(gene_record.input_nodes)
        output_nodes = []
        for i in range(len(input_nodes)):
            node_values[input_nodes[i]] = input_values[i]

        for node in individual.node_order:
            incoming_edges = individual.incoming_edges_map[node]
            if incoming_edges:
                parent_node_values = tuple((node_values[gene_record.edge_definitions[edge][0]],
                                            individual.edge_weights[edge]) for edge in incoming_edges)
                node_values[node] = max(0, np.sum(np.prod(parent_node_values, axis=1)))
            else:
                node_values[node] = 0

        for node in sorted(gene_record.output_nodes):
            incoming_edges = individual.incoming_edges_map[node]
            if incoming_edges:
                parent_node_values = tuple((node_values[gene_record.edge_definitions[edge][0]],
                                            individual.edge_weights[edge]) for edge in incoming_edges)
                output_nodes.append(np.sum(np.prod(parent_node_values, axis=1)))
            else:
                output_nodes.append(0)
        return output_nodes


class OnePlayerPlaySession(AbstractPlaySession):
    def __init__(self, individual, gene_record, game_instance):
        super().__init__(game_instance)
        self.individual = individual
        self.gene_record = gene_record
        self.game_instance = game_instance
        self.state_size = np.size(game_instance.get_state())
        self.inputs = [0] * self.state_size + [1]

    def _play_turn(self):
        state = self.game_instance.get_state()
        self.inputs[:self.state_size] = np.reshape(state, newshape=[self.state_size])

        predictions = self._feedforward(self.individual, self.inputs, self.gene_record)
        action = np.argmax(predictions)
        self.game_instance.do_action(action)


class TwoPlayerPlaySession(AbstractPlaySession):
    def __init__(self, individual_1, individual_2, gene_record_1, gene_record_2, game_instance, comms_size):
        super().__init__(game_instance)
        self.individual_1 = individual_1
        self.individual_2 = individual_2
        self.gene_record_1 = gene_record_1
        self.gene_record_2 = gene_record_2
        self.state_size = np.size(game_instance.get_state()[0])
        self.comms_size = comms_size
        blank_input = [0] * (self.state_size + comms_size) + [1]
        self.inputs = (blank_input, blank_input.copy())

    def _play_turn(self):
        inputs_1, inputs_2 = self.inputs
        states = self.game_instance.get_state()
        inputs_1[:self.state_size], inputs_2[:self.state_size] = np.reshape(states, newshape=[2, self.state_size])

        outputs_1 = self._feedforward(self.individual_1, inputs_1, self.gene_record_1)
        outputs_2 = self._feedforward(self.individual_2, inputs_2, self.gene_record_2)
        predictions_1 = outputs_1[:self.state_size]
        predictions_2 = outputs_2[:self.state_size]
        action = (np.argmax(predictions_1), np.argmax(predictions_2))
        self.game_instance.do_action(action)

        comms_1_to_2 = outputs_1[-self.comms_size:]
        comms_2_to_1 = outputs_2[-self.comms_size:]
        inputs_1[self.state_size:-1] = comms_2_to_1
        inputs_2[self.state_size:-1] = comms_1_to_2

