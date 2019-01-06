import numpy as np

from .play_models import TwoPlayerPlaySession, OnePlayerPlaySession
from .tf import Graph


def generate_one_player_game_fitness_function(game_instances, turn_limit=None):
    return lambda population: __one_player_game_fitness_function(game_instances, population, turn_limit)


def generate_two_player_game_fitness_function(game_instances, communication_size, turn_limit=None):
    return lambda population_1, population_2: __two_player_game_fitness_function(game_instances, communication_size, population_1, population_2, turn_limit)


def generate_two_player_game_fitness_function_tf(game_instances, communication_size, turn_limit=None, batch_size=None):
    container = __GameFitnessFunctionTfPersistenceContainer(game_instances, communication_size, turn_limit, batch_size)
    return lambda population_1, population_2: container.game_fitness_function_tf(population_1, population_2)


def __two_player_game_fitness_function(game_instances, comms_size, population_1, population_2, turn_limit=None):
    scores = []

    for individual_1 in population_1.individuals:
        scores.append([])
        for individual_2 in population_2.individuals:
            overall_score = 0

            for game_instance in game_instances:
                play_session = TwoPlayerPlaySession(individual_1, individual_2, population_1.gene_record,
                                                    population_2.gene_record, game_instance, comms_size)
                if turn_limit is None:
                    play_session.play_until_termination()
                else:
                    play_session.play_turns(turn_limit)

                overall_score += play_session.get_current_score()
                game_instance.reset()

            overall_score /= len(game_instances)
            scores[-1].append(overall_score)
    return scores


def __one_player_game_fitness_function(game_instances, population, turn_limit=None):
    scores = []

    for individual in population.individuals:
        overall_score = 0
        for game_instance in game_instances:
            play_session = OnePlayerPlaySession(individual, population.gene_record, game_instance)

            if turn_limit is None:
                play_session.play_until_termination()
            else:
                play_session.play_turns(turn_limit)

            overall_score += game_instance.get_score()
            game_instance.reset()

        overall_score /= len(game_instances)
        scores.append(overall_score)
    return scores


class __GameFitnessFunctionTfPersistenceContainer:

    def __init__(self, game_instances, communication_size, turn_limit=None, batch_size=None):
        self.game_instances = game_instances
        self.communication_size = communication_size
        self.turn_limit = turn_limit
        self.batch_size = batch_size
        self.graph_1 = None
        self.graph_2 = None
        self.n_games = len(self.game_instances)
        self.state_size = np.size(self.game_instances[0].get_state()[0])
        self.empty_inputs = [0] * (self.state_size + self.communication_size) + [1]

    def __generate_empty_inputs(self, population_1_size, population_2_size):
        return [[self.empty_inputs.copy() for _ in range(population_2_size) for _ in self.game_instances] for _ in range(population_1_size)], [[self.empty_inputs.copy() for _ in range(population_1_size) for _ in self.game_instances] for _ in range(population_2_size)]

    def game_fitness_function_tf(self, population_1, population_2, ):
        if self.graph_1 is None:
            self.graph_1 = Graph(population_1.gene_record)
            self.graph_2 = Graph(population_2.gene_record)
            Graph.start_session()
        else:
            self.graph_1.update_structure()
            self.graph_2.update_structure()

        inputs_1_buffer, inputs_2_buffer = self.__generate_empty_inputs(population_1.size, population_2.size)
        feed_dict_1 = self.graph_1.generate_feed_dict(population_1.individuals, inputs_1_buffer)
        feed_dict_2 = self.graph_2.generate_feed_dict(population_2.individuals, inputs_2_buffer)
        active_games = [(i, j, k, self.game_instances[k].clone()) for i in range(population_1.size) for j in range(population_2.size) for k in range(self.n_games)]
        scores = [[[] for _ in range(population_2.size)] for _ in range(population_1.size)]

        current_turn = 0
        while active_games:
            current_turn += 1

            # Copy states into inputs
            for active_game in active_games:
                individual_1, individual_2, game_id, game_instance = active_game
                index_1 = individual_2 * self.n_games + game_id
                index_2 = individual_1 * self.n_games + game_id
                states = game_instance.get_state()
                (inputs_1_buffer[individual_1][index_1][:self.state_size],
                 inputs_2_buffer[individual_2][index_2][:self.state_size]) = np.reshape(states, newshape=[2, -1])

            # Calculate outputs
            if self.batch_size is None:
                outputs_1 = self.graph_1.feed_forward(feed_dict_1)
                outputs_2 = self.graph_2.feed_forward(feed_dict_2)
            else:
                index = 0
                item_count_1 = len(inputs_1_buffer)
                item_count_2 = len(inputs_2_buffer)
                outputs_1 = None
                outputs_2 = None
                while index < item_count_1 and index < item_count_2:
                    batch_feed_dict_1 = {placeholder: feed_dict_1[placeholder][index:index + self.batch_size]
                                         for placeholder in feed_dict_1}
                    batch_feed_dict_2 = {placeholder: feed_dict_2[placeholder][index:index + self.batch_size]
                                         for placeholder in feed_dict_2}
                    batch_outputs_1 = self.graph_1.feed_forward(batch_feed_dict_1)
                    batch_outputs_2 = self.graph_2.feed_forward(batch_feed_dict_2)
                    outputs_1 = batch_outputs_1 if outputs_1 is None else np.concatenate((outputs_1, batch_outputs_1))
                    outputs_2 = batch_outputs_2 if outputs_2 is None else np.concatenate((outputs_2, batch_outputs_2))
                    index += self.batch_size

            # Do actions and copy comms
            for active_game in active_games:
                individual_1, individual_2, game_id, game_instance = active_game
                index_1 = individual_2 * self.n_games + game_id
                index_2 = individual_1 * self.n_games + game_id

                # Do action
                predictions_1 = outputs_1[individual_1][index_1][:-self.communication_size]
                predictions_2 = outputs_2[individual_2][index_2][:-self.communication_size]
                action_1 = np.argmax(predictions_1)
                action_2 = np.argmax(predictions_2)
                game_instance.do_action((action_1, action_2))

                if not (game_instance.is_terminated() or (self.turn_limit is not None and current_turn >= self.turn_limit)):
                    # Copy comms into inputs
                    comms_1_to_2 = outputs_1[individual_1][index_1][-self.communication_size:]
                    comms_2_to_1 = outputs_2[individual_2][index_2][-self.communication_size:]
                    inputs_1_buffer[individual_1][index_1][self.state_size:-1] = comms_2_to_1
                    inputs_2_buffer[individual_2][index_2][self.state_size:-1] = comms_1_to_2
                else:
                    # Game finished
                    scores[individual_1][individual_2].append(game_instance.get_score())
                    active_games.remove(active_game)
        for individual_1 in range(len(scores)):
            for individual_2 in range(len(scores[individual_1])):
                scores[individual_1][individual_2] = sum(scores[individual_1][individual_2]) / self.n_games

        return scores