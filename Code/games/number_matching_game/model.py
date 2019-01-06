from random import Random

random = Random()


class NumberMatchingGameModel:
    components_per_option = 2

    def __init__(self, option_count, solution_option_count):
        self.option_count = option_count
        self.solution_option_count = solution_option_count
        self.options_1 = None
        self.options_2 = None
        self.solution = None
        self.solution_options_1 = None
        self.solution_options_2 = None

        self.component_count = self.option_count * self.components_per_option + 1
        self.components = [1 << i for i in range(self.component_count)]
        self.new_random_options()

    def new_random_options(self):
        random.shuffle(self.components)
        self.solution = 0

        solution_component_count = self.solution_option_count * self.components_per_option
        solution_components = self.components[:solution_component_count]
        non_solution_components = self.components[solution_component_count:]

        for component in solution_components:
            self.solution += component

        self.solution_options_1, self.solution_options_2 = self.options_from_components(solution_components)
        non_solution_options_1, non_solution_options_2 = self.options_from_components(non_solution_components)

        self.options_1 = self.solution_options_1 + non_solution_options_1
        self.options_2 = self.solution_options_2 + non_solution_options_2

        random.shuffle(self.options_1)
        random.shuffle(self.options_2)

    def options_from_components(self, components):
        offset = random.choice(range(1, self.components_per_option))
        options_1 = []
        options_2 = []
        i = 0
        for _ in range(len(components) // self.components_per_option):
            option_1 = 0
            option_2 = 0
            for _ in range(self.components_per_option):
                option_1 += components[i]
                option_2 += components[(i + offset) % len(components)]
                i += 1
            options_1.append(option_1)
            options_2.append(option_2)
        return options_1, options_2

    def get_sums(self, selection_1, selection_2):
        sum_1 = 0
        sum_2 = 0
        for i in range(self.option_count):
            if selection_1[i]:
                sum_1 += self.options_1[i]
            if selection_2[i]:
                sum_2 += self.options_2[i]
        return sum_1, sum_2

    def get_score(self, selection_1, selection_2):
        if not (selection_1 or selection_2):
            return 1

        sum_1, sum_2 = self.get_sums(selection_1, selection_2)
        if not (sum_1 or sum_2):
            return 1
        matching = 2 * bin(sum_1 & sum_2).count("1")
        non_matching = bin(sum_1 ^ sum_2).count("1")
        return 1 + 99 * matching / (matching + non_matching)
