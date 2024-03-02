from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from eval import intrinsic_value, interaction_value
from data import CAT, NUM_SUBCATS, INTERACTION_MODEL, CHARACTERISTIC_DISTANCE, LAND_DIMENSION, POPULATION_SIZE, \
    MAX_GENERATION, MUTATION_RATE, CROSSOVER_RATE, ELITISM_RATE, INTERACTION_COEFFICIENT, ROULETTE_RATE, COLORS_DICT, \
    preamble, post

from time import time
import os
from scipy.spatial.distance import pdist

np.set_printoptions(suppress=True)

os.makedirs('table', exist_ok=True)


def find_group_borders(arr):
    diff = np.diff(arr.T, axis=0)
    vlines = {g: np.where(d != 0)[0] + 1 for g, d in zip(range(2, arr.shape[1] + 1), diff)}

    diff_t = np.diff(arr, axis=0)
    hlines = {g: np.where(d != 0)[0] + 1 for g, d in zip(range(2, arr.shape[0] + 1), diff_t)}

    vline_str = ', '.join(
        [f'vline{{{k}}}={{{",".join(map(str, v))}}}{{black}}' for k, v in vlines.items() if v.size > 0]) + ', '
    hline_str = ', '.join(
        [f'hline{{{k}}}={{{",".join(map(str, v))}}}{{black}}' for k, v in hlines.items() if v.size > 0]) + ', '

    return vline_str + hline_str + '\n'


def color_cat(cat):
    arr = cat.copy()
    arr = np.pad(arr, ((0, 0), (1, 1)), mode='constant', constant_values=-1)
    diff = np.diff(arr, axis=1)
    colors = {g: (np.where(d != 0)[0], arr[i][np.where(d != 0)[0] + 1]) for i, (g, d) in
              enumerate(zip(range(1, arr.shape[0] + 1), diff))}
    col_str = ''
    for k, (idx, val) in colors.items():
        col_str += ', '.join(
            [
                f'{f"vline{{{idx[i] + 2}-{idx[i + 1]}}}={{{k}}}{{{COLORS_DICT[val[i] % len(COLORS_DICT)]}}}," if idx[i] + 2 <= idx[i + 1] else ""}'
                f'cell{{{k}}}{{{idx[i] + 1}-{idx[i + 1]}}}={{}}{{{COLORS_DICT[val[i] % len(COLORS_DICT)]}}}'
                for i in range(len(idx) - 1)
            ]
        ) + ', '

    arr = cat.copy()
    arr = np.pad(arr, ((1, 1), (0, 0)), mode='constant', constant_values=-1)
    diff_t = np.diff(arr.T, axis=1)
    colors = {g: (np.where(d != 0)[0], arr.T[i][np.where(d != 0)[0] + 1]) for i, (g, d) in
              enumerate(zip(range(1, arr.shape[1] + 1), diff_t))}
    for k, (idx, val) in colors.items():
        col_str += ''.join(
            [
                f'hline{{{idx[i] + 2}-{idx[i + 1]}}}={{{k}}}{{{COLORS_DICT[val[i] % len(COLORS_DICT)]}}}, '
                if idx[i] + 2 <= idx[i + 1] else "" for i in range(len(idx) - 1)
            ])
    return col_str + '\n'


class UrbanPlanning:
    print(intrinsic_value(0, 0, 1))

    def __init__(self, land_dimension, population_size, max_generation, mutation_rate, crossover_rate, elitism_rate,
                 roulette_rate):
        self.land_dimension = land_dimension
        self.population_size = population_size
        self.max_generation = max_generation
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.roulette_rate = roulette_rate

    def initialize(self):
        population = []
        cats = np.random.randint(0, len(CAT), (self.population_size, *self.land_dimension))
        subcats = np.array([np.random.randint(0, NUM_SUBCATS[c], self.land_dimension) for c in cats])
        group = np.arange(1, self.land_dimension[0] * self.land_dimension[1] + 1).reshape(*self.land_dimension)
        group_two_two = (np.arange(1, self.land_dimension[0] * self.land_dimension[1] + 1).reshape(
            *self.land_dimension) + 1) // 2
        population = np.array(
            [{'cat': cats[i], 'subcat': subcats[i], 'group': group} for i in range(self.population_size)])
        return population

    def intrinsic(self, individual):
        unique_groups = np.unique(individual['group'])
        unique_groups = unique_groups[unique_groups != 0]
        groups = [np.argwhere(individual['group'] == i).tolist() for i in unique_groups]
        group_cats = np.array([individual['cat'][groups[i][0][0], groups[i][0][1]] for i in range(len(groups))])
        group_subcats = np.array([individual['subcat'][groups[i][0][0], groups[i][0][1]] for i in range(len(groups))])
        group_sizes = np.array([len(groups[i]) for i in range(len(groups))])
        intrinsic_values = intrinsic_value(group_cats, group_subcats, group_sizes)
        individual['groups'] = groups
        individual['group_cats'] = group_cats
        individual['group_subcats'] = group_subcats
        individual['group_sizes'] = group_sizes
        individual['intrinsic_values'] = intrinsic_values
        intrinsic_value_sum = np.sum(intrinsic_values)
        return intrinsic_value_sum

    def interaction(self, individual):
        groups = individual['groups']
        group_cats = individual['group_cats']
        group_subcats = individual['group_subcats']
        group_sizes = individual['group_sizes']
        intrinsics = individual['intrinsic_values']
        group_mean_positions = np.array([np.mean(groups[i], axis=0) for i in range(len(groups))])
        dists = pdist(group_mean_positions)
        upper_tri_indices = np.triu_indices(len(groups), 1)
        char_dists = CHARACTERISTIC_DISTANCE[group_cats[upper_tri_indices[0]], group_cats[upper_tri_indices[1]]]
        coefficients = INTERACTION_COEFFICIENT[group_cats[upper_tri_indices[0]], group_cats[upper_tri_indices[1]]]
        models = INTERACTION_MODEL[group_cats[upper_tri_indices[0]], group_cats[upper_tri_indices[1]]]
        intrinsics = np.sqrt(intrinsics[upper_tri_indices[0]] * intrinsics[upper_tri_indices[1]])
        interaction_values = interaction_value(dists / 10, char_dists, models) * coefficients * intrinsics
        individual['interaction_values'] = interaction_values
        interaction_value_sum = np.sum(interaction_values) / np.sqrt(len(groups))
        return interaction_value_sum

    def fitness(self, individuals):
        intrinsic_values = np.vectorize(self.intrinsic, otypes=[np.float64])(individuals)
        interaction_values = np.vectorize(self.interaction, otypes=[np.float64])(individuals)
        individual_values = intrinsic_values + interaction_values
        # if num_cat is not 6, make the individual value 0
        num_cats = np.vectorize(lambda x: len(np.unique(x['cat'])))(individuals)
        individual_values[num_cats != 6] = 0
        return individual_values

    def selection(self, population):
        fitness = self.fitness(population)
        sorted_indices = np.argsort(fitness)[::-1]
        population = population[sorted_indices]
        num_elites = int(self.population_size * self.elitism_rate)
        elites = population[:num_elites]
        fitness = fitness[num_elites:]
        population = population[num_elites:]
        fitness = np.nan_to_num(fitness)
        fitness = fitness - np.min(fitness)
        fitness = np.nan_to_num(fitness)
        if np.sum(fitness) == 0:
            relative_fitness = np.ones_like(fitness) / len(fitness)
        else:
            relative_fitness = fitness / np.sum(fitness)
        num_roulette = int(self.population_size * self.roulette_rate)
        chosen_indices = np.random.choice(len(population), size=num_roulette,
                                          p=relative_fitness)
        selected = np.array([population[i] for i in chosen_indices])
        selected = np.concatenate((elites, selected))
        return selected

    def crossover(self, population):
        num_crossover = int(self.population_size * self.crossover_rate)
        num_parents = len(population)
        parents = population
        children = []
        for i in range(num_crossover):
            parent1 = parents[np.random.randint(0, num_parents)]
            parent2 = parents[np.random.randint(0, num_parents)]
            child = parent1.copy()
            num_group1 = len(parent1['groups'])
            num_group2 = len(parent2['groups'])
            group1 = np.random.choice(num_group1, size=int(num_group1 / 2), replace=False)
            group2 = np.random.choice(num_group2, size=int(num_group2 / 2), replace=False)
            pos1 = np.round(np.array([np.mean(parent1['groups'][i], axis=0) for i in group1]))
            pos2 = np.round(np.array([np.mean(parent2['groups'][i], axis=0) for i in group2]))
            group_sizes1 = parent1['group_sizes']
            group_sizes2 = parent2['group_sizes']
            group_cats1 = parent1['group_cats']
            group_cats2 = parent2['group_cats']
            group_subcats1 = parent1['group_subcats']
            group_subcats2 = parent2['group_subcats']
            # prin
            mean_pos = np.int64(np.vstack((pos1, pos2)))
            mean_pos, unique_indices = np.unique(mean_pos, axis=0, return_index=True)
            group_sizes = np.concatenate((group_sizes1, group_sizes2), axis=0)[unique_indices]
            group_cats = np.concatenate((group_cats1, group_cats2), axis=0)[unique_indices]
            group_subcats = np.concatenate((group_subcats1, group_subcats2), axis=0)[unique_indices]
            group_sizes = np.concatenate(([0], group_sizes))
            group_cats = np.concatenate(([0], group_cats))
            group_subcats = np.concatenate(([0], group_subcats))
            board = np.zeros_like(parent1['group'], dtype=np.int64)
            board[mean_pos[:, 0], mean_pos[:, 1]] = np.arange(1, len(mean_pos) + 1)

            # while there are any zero cells in board
            while np.any(board == 0):
                # We expand the groups to its adjacent four cells. Store the group number in expand
                expand = np.empty_like(board, dtype=object)
                expand.fill([])
                dxy = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])
                pad_board = np.pad(board, ((1, 2), (1, 2)), mode='constant')
                adj_boards = np.dstack(np.array([pad_board[1 + dx:-2 + dx, 1 + dy:-2 + dy] for dx, dy in dxy]))
                expand = np.array(
                    [adj_boards[i, j][adj_boards[i, j] != 0].tolist() for i, j in np.ndindex(board.shape)],
                    dtype=object).reshape(board.shape)
                expand[board != 0] = np.array(None, dtype=object)

                def random_select(x):
                    # choose random element from x with probability proportional to group_sizes
                    p = group_sizes[x]
                    p = p / np.sum(p)
                    return np.random.choice(x, p=p) if x else 0

                board += np.vectorize(random_select, otypes=[np.int64])(expand)

            child['group'] = board
            # set the subcat and cat of the child according to the group
            child['cat'] = group_cats[board]
            child['subcat'] = group_subcats[board]
            children.append(child)
        children = np.array(children)
        return children

    def mutation(self, population):
        return population

    def plot(self, individual):
        # I will draw three subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        # show the groups of the individual, each group has a different color
        # the categories of each group are shown inside as a number inside the heatmap box
        ax1.imshow(individual['group'], cmap='tab20')
        for (i, j), z in np.ndenumerate(individual['group']):
            if z != 0:
                ax1.text(j, i, z, ha='center', va='center', color='w')
        ax1.title.set_text('Groups')
        # show the categories of the individual
        ax2.imshow(np.int32(individual['cat']), cmap='tab20')

        for (i, j), z in np.ndenumerate(individual['cat']):
            ax2.text(j, i, z, ha='center', va='center', color='w')
        cats = np.int32(np.unique(individual['cat']))
        # show legend for each color as a category, and locate it at the outside top right of the plot
        cat_list = [CAT[c] for c in cats]
        ax2.legend([plt.Rectangle((0, 0), 1, 1, fc=cm) for cm in (
            plt.cm.tab20.colors[0], plt.cm.tab20.colors[4], plt.cm.tab20.colors[8], plt.cm.tab20.colors[13],
            plt.cm.tab20.colors[16], plt.cm.tab20.colors[19])], cat_list, bbox_to_anchor=(1.3, 1.2))
        ax2.title.set_text('Categories')
        plt.show()

    def run(self):
        st = time()
        best_score = 0
        population = self.initialize()
        print("max:", self.fitness(population).max())
        best_score = self.fitness(population).max()
        self.plot(population[0])
        for i in tqdm(range(self.max_generation), desc="Generation"):
            parents = self.selection(population)
            children = self.crossover(parents)
            children = self.mutation(children)
            population = np.concatenate((parents, children))
            values = self.fitness(population)
            best_index = np.argmax(values)
            if values[best_index] > best_score:
                best_score = values[best_index]
                print("max:", values.max(), "min:", values.min(), "mean:", values.mean())
                self.plot(population[best_index])
        values = self.fitness(population)
        print("max:", values.max(), "min:", values.min(), "mean:", values.mean())
        best_index = np.argmax(values)
        print([population[best_index][i] for i in ['cat', 'subcat', 'group']])
        self.plot(population[best_index])
        print(time() - st)
        self.draw_table(population[best_index]['cat'], population[best_index]['group'])

    def draw_table(self, cat, group):
        table = preamble
        table += "\\begin{document}\n"
        table += "\\begin{tblr}{\n"
        table += "rowsep=0pt, colsep=0pt, columns={5mm,c}, rows={5mm,c},\n"
        table += "vline{1,Z}={1-Z}{solid},hline{1,Z}={1-Z}{solid},\n"
        table += color_cat(cat)
        table += find_group_borders(group)
        table += "}\n"
        # Make the table & symbols in size of land_dimension
        table += (" & " * (self.land_dimension[1] - 1) + "\\\\\n") * (self.land_dimension[0])
        table += "\\end{tblr}\n"
        table += post
        table += "\\end{document}\n"
        print(table)
        # Write to a file
        with open('table/table.tex', 'w') as f:
            f.write(table)
        # Compile the LaTeX file
        os.chdir('table')
        os.system("pdflatex table.tex")


if __name__ == '__main__':
    np.random.seed(42)
    ga = UrbanPlanning(LAND_DIMENSION, POPULATION_SIZE, MAX_GENERATION, MUTATION_RATE, CROSSOVER_RATE, ELITISM_RATE,
                       ROULETTE_RATE)
    ga.run()
