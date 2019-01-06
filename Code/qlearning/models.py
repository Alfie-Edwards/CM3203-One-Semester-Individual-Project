import math
from random import Random
import numpy as np
from . import QMatrix

FUTURE_REWARD_DECAY = 0.95

random = Random()
output = True


class SingleAgentModel:

    def __init__(self, agent):
        self.agent = agent
        self.n_actions = agent.n_actions
        self.qmatrix = QMatrix(self.n_actions)

    def gather_experience_from_games(self, instances, initial_prediction_chance, prediction_chance_decay, turn_limit=None):
        replay_memory = {instance: [] for instance in instances}
        active_instances = list(instances)
        prediction_chances = [initial_prediction_chance for _ in instances]
        turns = 0
        while active_instances:
            turns += 1
            instances_for_prediction = []
            states_for_prediction = []
            for i in range(len(active_instances)):
                instance = active_instances[i]
                if random.random() >= prediction_chances[i]:
                    action = random.choice(range(self.n_actions))
                    state = instance.get_state()
                    score = instance.get_score()
                    instance.do_action(action)
                    delta_score = instance.get_score() - score
                    replay_memory[instance].append((state, action, delta_score))
                else:
                    instances_for_prediction.append(instance)
                    states_for_prediction.append(instance.get_state())
                    prediction_chances[i] *= (1 - prediction_chance_decay)
            predicted_best_actions = self.agent.predict_best_actions(states_for_prediction)
            for i in range(len(instances_for_prediction)):
                instance = instances_for_prediction[i]
                action = predicted_best_actions[i]
                state = states_for_prediction[i]
                score = instance.get_score()
                instance.do_action(action)
                delta_score = instance.get_score() - score
                replay_memory[instance].append((state, action, delta_score))

            i = 0
            while i < len(active_instances):
                if active_instances[i].is_terminated() or (turn_limit is not None and turns >= turn_limit):
                    active_instances.pop(i)
                    prediction_chances.pop(i)
                    complete = len(instances) - len(active_instances)
                    print("Games complete:", complete, "/", len(instances), ("(" + str(turns)), "turns)")
                else:
                    i += 1

        for instance in instances:
            replay = replay_memory[instance]
            prev_best_outcome = 0
            for i in range(len(replay) - 1, -1, -1):
                state, action, outcome = replay[i]
                outcome += max(prev_best_outcome, 0) * FUTURE_REWARD_DECAY
                self.qmatrix.update(state, action, outcome)
                prev_best_outcome = np.max(self.qmatrix.get(state))

    def train_agent_from_experiences(self, n):
        n = min(n, len(self.qmatrix))
        states, outcomes, mask = self.qmatrix.get_random_samples(n)

        loss = self.agent.train(states, outcomes, mask)
        accuracy = math.e ** (-loss)
        return accuracy

    def predict_best_actions(self, states):
        return self.agent.predict_best_actions(states)


class DoubleAgentModel:

    def __init__(self, agent):
        self.agent = agent
        self.n_actions = agent.n_actions
        self.qmatrix = QMatrix(self.n_actions * 2)

    def gather_experience_from_games(self, instances, initial_prediction_chance, prediction_chance_decay, turn_limit=None):
        replay_memory = {instance: [] for instance in instances}
        active_instances = list(instances)
        prediction_chances = [initial_prediction_chance for _ in instances]
        turns = 0
        while active_instances:
            turns += 1
            instances_for_prediction = []
            states_for_prediction = []
            for i in range(len(active_instances)):
                instance = active_instances[i]
                if random.random() >= prediction_chances[i]:
                    action = (random.choice(range(self.n_actions)), random.choice(range(self.n_actions)))
                    state = instance.get_state()
                    score = instance.get_score()
                    instance.do_action(action)
                    delta_score = instance.get_score() - score
                    replay_memory[instance].append((state, action, delta_score))
                else:
                    instances_for_prediction.append(instance)
                    states_for_prediction.append(instance.get_state())
                    prediction_chances[i] *= (1 - prediction_chance_decay)
            predicted_best_actions, _, _ = self.agent.predict_best_actions(states_for_prediction)
            for i in range(len(instances_for_prediction)):
                instance = instances_for_prediction[i]
                action = predicted_best_actions[i]
                state = states_for_prediction[i]
                score = instance.get_score()
                instance.do_action(action)
                delta_score = instance.get_score() - score
                replay_memory[instance].append((state, action, delta_score))

            i = 0
            while i < len(active_instances):
                if active_instances[i].is_terminated() or (turn_limit is not None and turns >= turn_limit):
                    active_instances.pop(i)
                    prediction_chances.pop(i)
                    complete = len(instances) - len(active_instances)
                    print("Games complete:", complete, "/", len(instances), ("(" + str(turns)), "turns)")
                else:
                    i += 1

        for instance in instances:
            replay = replay_memory[instance]
            prev_best_outcome = 0
            for i in range(len(replay) - 1, -1, -1):
                state, action, outcome = replay[i]
                outcome += max(prev_best_outcome, 0) * FUTURE_REWARD_DECAY
                self.qmatrix.update(state, action[0], outcome)
                self.qmatrix.update(state, self.n_actions + action[1], outcome)
                prev_best_outcome = np.max(self.qmatrix.get(state))

    def train_agent_from_experiences(self, n):
        n = min(n, len(self.qmatrix))
        state_pairs, outcomes, mask = self.qmatrix.get_random_samples(n)
        outcome_pairs = []
        outcome_pairs_mask = []
        for i in range(len(outcomes)):
            outcome_pairs.append((outcomes[i][:self.n_actions], outcomes[i][self.n_actions:]))
            outcome_pairs_mask.append((mask[i][:self.n_actions], mask[i][self.n_actions:]))
        loss = self.agent.train(state_pairs, outcome_pairs, outcome_pairs_mask)
        accuracy = math.e ** (-loss)
        return accuracy

    def predict_best_actions(self, states):
        return self.agent.predict_best_actions(states)
