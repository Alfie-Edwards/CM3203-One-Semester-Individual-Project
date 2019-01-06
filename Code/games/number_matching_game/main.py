import itertools

from . import NumberMatchingGameModel


def debug_model(model):
    options_1, options_2, solution = model.generate_options()
    print("First player options:", options_1)
    print("Second player options:", options_2)
    print("Solution:",  solution)

    sums_1 = []
    for x in [itertools.combinations(options_1, n) for n in range(len(options_1) + 1)]:
        for y in x:
            sums_1.append(sum(y))
    sums_2 = []
    for x in [itertools.combinations(options_2, n) for n in range(len(options_2) + 1)]:
        for y in x:
            sums_2.append(sum(y))

    print("\nFirst player combinations:", sums_1)
    print("Second player combinations:", sums_2)
    matches = []
    for sum_1 in sums_1:
        for sum_2 in sums_2:
            if sum_1 == sum_2:
                matches.append(sum_1)
    print("Matching combinations:", matches)


model = NumberMatchingGameModel(5, 2)
debug_model(model)
