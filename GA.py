import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 5  # DNA length
POP_SIZE = 10  # population size
CROSS_RATE = 0.8  # mating probability (DNA crossover)
MUTATION_RATE = 0.0003  # mutation probability
N_GENERATIONS = 50
X_BOUND = [-1, 4.9]  # x upper and lower bounds
pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))  # initialize the pop DNA


def F(x):
    return x ** 2 - 3 * x + 2


def get_fitness(pred):
    return pred + 1e-3 - np.min(pred)  # - 防止fitness是负数  e-3防止sum为0


def translateDNA(pop):
    return pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * X_BOUND[1]


def select(pop, fitness):  # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness / fitness.sum())
    return pop[idx]


def crossover(parent, pop):  # mating process (genes crossover)
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)  # select another individual from pop
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)  # choose crossover points
        parent[cross_points] = pop[i_, cross_points]  # mating and produce one child
    return parent


def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child


pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))  # initialize the pop DNA
# plt.ion()  # something about plotting
# x = np.linspace(*X_BOUND, 200)
# plt.plot(x, F(x))
plt.ion()  # something about plotting
x = np.linspace(*X_BOUND, 200)
plt.plot(x, F(x))
for _ in range(N_GENERATIONS):
    F_values = F(translateDNA(pop))  # compute function value by extracting DNA

    # something about plotting
    if 'sca' in globals(): sca.remove()
    sca = plt.scatter(translateDNA(pop), F_values, s=200, lw=0, c='red', alpha=0.5);
    plt.pause(0.05)

    # GA part (evolution)
    fitness = get_fitness(F_values)
    print("Most fitted DNA: ", pop[np.argmax(fitness), :])
    print("十进制", translateDNA(pop))
    pop = select(pop, fitness)
    pop_copy = pop.copy()
    for parent in pop:
        child = crossover(parent, pop_copy)
        child = mutate(child)
        parent[:] = child  # parent is replaced by its child

plt.ioff();
plt.show()
