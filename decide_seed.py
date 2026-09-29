import random

random.seed(42)
seeds = sorted(
    random.sample(range(1, 201), 5)
)

print(seeds)
# [7, 29, 71, 164, 190]