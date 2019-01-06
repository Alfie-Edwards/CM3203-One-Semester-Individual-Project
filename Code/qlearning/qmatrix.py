from ast import literal_eval
from random import Random


random = Random()


class QMatrix:

    NEGATIVE_INFINITY = float("-inf")
    DEFAULT_VALUE = -1

    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.values = {}
        self.entry_map = {}

    def __len__(self):
        return len(self.values)

    def get(self, state, action=None):
        if state not in self.entry_map:
            return [self.DEFAULT_VALUE] * self.n_actions if action is None else self.DEFAULT_VALUE
        else:
            if action is None:
                return self.values[state]
            else:
                return self.values[state][action]

    def update(self, state, action, outcome):
        if state not in self.entry_map:
            self.values[state] = [self.DEFAULT_VALUE] * self.n_actions
            self.entry_map[state] = [0] * self.n_actions
        outcome += self.values[state][action] * self.entry_map[state][action]
        self.entry_map[state][action] += 1
        outcome /= self.entry_map[state][action]
        self.values[state][action] = outcome

    def get_random_samples(self, n):
        states = random.sample(self.values.keys(), n)
        outcomes = tuple(self.values[state] for state in states)
        mask = tuple(tuple(count > 0 for count in self.entry_map[state]) for state in states)
        return states, outcomes, mask

    def __str__(self):
        output = []
        for state in self.values:
            output.append(str(state))
            output.append("~")
            output.append(str(self.values[state]))
            output.append("\n")
        return ''.join(output)

    @staticmethod
    def from_string(qmatrix_string):
            qmatrix = None
            for line in qmatrix_string.splitlines():
                if "~" not in line:
                    continue
                sep_index = line.index("~")
                state = literal_eval(line[:sep_index])
                outcomes = literal_eval(line[sep_index + 1:])
                entry_map = [True for _ in range(len(outcomes))]
                for i in range(len(outcomes)):
                    if outcomes[i] == QMatrix.DEFAULT_VALUE:
                        entry_map[i] = False

                if qmatrix is None:
                    qmatrix = QMatrix(len(outcomes))
                qmatrix.values[state] = outcomes
                qmatrix.entry_map[state] = entry_map
            return qmatrix
