import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import multiprocessing

"""
@author
11612126 李可明
"""

def build_adjcent_list(distance_matrix):
    adjcent_dict_list = dict()
    for i in range(CITIES_NUM):
        adjcent_dict_list[i] = sorted(list(range(CITIES_NUM)), key=lambda index:distance_matrix[i][index])[1:]
        # remove itself
    return adjcent_dict_list


def cal_distance(c1, c2):
    tmp = c1 - c2
    return (tmp[0] * tmp[0] + tmp[1] * tmp[1]) ** 0.5


def generate_dis_matrix(loc_raw_data):
    """

    :return:
    """
    dis = np.zeros(shape=(CITIES_NUM, CITIES_NUM))
    for i in range(CITIES_NUM):
        for j in range(i+1, CITIES_NUM):
            dis[i][j] = cal_distance(loc_raw_data[i], loc_raw_data[j])
            dis[j][i] = dis[i][j]
    return dis


def draw(data_index):
    data = raw_data[data_index].transpose()
    def update_line(num, data, line):
        line.set_data(data[..., :num])
        return line,

    fig1 = plt.figure()

    data = np.hstack((data, [[data[0, 0]], [data[1, 0]]]))
    l, = plt.plot([], [], 'r-')
    plt.xlim(0, 1000)
    plt.ylim(0, 1000)
    plt.xlabel('x')
    plt.title('test')
    line_ani = animation.FuncAnimation(fig1, update_line, 110, fargs=(data, l),
                                       interval=10, blit=True, repeat=False)
    # To save the animation, use the command: line_ani.save('lines.mp4')
    plt.show()


# ============================================Algorithm begin here========================================


def init_population_greedy(pop_size):
    pop = []
    for n in range(pop_size):
        i = individual()
        i.greedy_init(n)   # heuristic
        pop.append(i)
    return pop


def population_sort(population):
    return sorted(population)


def in_population(population, individual):
    # TODO
    return False


class individual:

    dis_m=None
    CITIES_NUM=100
    adj_list=None

    def __init__(self):
        self.gene = None
        self.fitness = 8000

    def greedy_init(self, start):
        visit_node = set()
        visit_node.add(start)
        ordered_list = [start]
        while len(ordered_list) < individual.CITIES_NUM:
            for aj_node in individual.adj_list[start]:
                if aj_node not in visit_node:
                    visit_node.add(aj_node)
                    ordered_list.append(aj_node)
                    start = aj_node
                    break
        self.gene = ordered_list
        self.fitness = self.cal_fitness()

    def cal_fitness(self):
        result = 0
        for i in range(len(self.gene)):
            c1 = self.gene[i-1]
            c2 = self.gene[i]
            result += individual.dis_m[c1][c2]
        return result

    def __lt__(self, other):
        return self.fitness < other.fitness

    def mutation(self, P_m, c_range):
        """

        :return: a shift mutation method
        """
        for i in range(len(self.gene)):
            if random.random() < P_m:
                r_num = self.gene.pop(i)
                self.gene.insert(random.randint(0, c_range), r_num)
        self.fitness = self.cal_fitness()


    @staticmethod
    def selfing(parent):
        point1, point2 = random.sample(range(individual.CITIES_NUM), 2)  # check
        if point1 > point2:
            point1, point2 = point2, point1
        tmp1 = parent[:point1]
        tmp2 = parent[point1:point2]
        tmp3 = parent[point2:]
        tmp = tmp1+tmp3
        point = np.random.randint(len(tmp))
        tmp1 = tmp[:point]
        tmp3 = tmp[point:]
        child = tmp1+tmp2+tmp3
        new_individual = individual()
        new_individual.gene = child
        new_individual.fitness = new_individual.cal_fitness()
        return new_individual


    @staticmethod
    def crossover(parent1, parent2):
        """
        order crossover: this function is no longer used
        """
        child = copy.deepcopy(parent1.gene)
        point1, point2 = random.sample(range(CITIES_NUM+1), 2)# check
        if point1 > point2:
            point1, point2 = point2, point1
        rm_set = set(child[point1: point2])
        if point1 > 0:
            index = 0
        else:
            index = point2
        for i in parent2.gene:
            if i not in rm_set:
                child[index] = i
                index += 1
                if index == point1:
                    index = point2
        new_individual = individual()
        new_individual.gene = child
        new_individual.fitness = new_individual.cal_fitness()
        return new_individual

    @staticmethod
    def invi_reverse(invi):
        """
        this  function is no longer used
        :param invi:
        :return:
        """
        point1, point2 = random.sample(range(CITIES_NUM), 2)  # check
        if point1 > point2:
            point1, point2 = point2, point1
        invi[point1:point2].reverse()
        new_individual = individual()
        new_individual.gene = invi
        new_individual.fitness = new_individual.cal_fitness()
        return new_individual


def genetic_algorithm(P_M, P_m, c_range, dis_matrix, aj_list):
    individual.adj_list=aj_list
    individual.dis_m = dis_matrix
    pop_size = 2
    population_list = init_population_greedy(100)
    population_list = population_sort(population_list)
    generation = 1
    pre_best = 8000
    start = time.time()
    while generation < 20000*25:
        best = population_list[0].fitness
        if best < pre_best:
            # plt.clf()
            # draw(population_list[0].gene)
            print(generation, best)
            pre_best = best
            print(population_list[0].gene)
        for i in range(pop_size):
            # binary tournament
            # ps = random.sample(range(pop_size), 4)
            # parents = sorted(ps)
            # parent1 = population_list[parents[0]]
            # parent2 = population_list[parents[1]]
            # child = individual.crossover(parent1, parent2)
            ps = random.sample(range(pop_size), 2)
            parent = population_list[sorted(ps)[0]]
            child = individual.selfing(copy.deepcopy(parent.gene))
            if child.fitness > best:
               if random.random() < P_M:
                   child.mutation(P_m, c_range)
            if not in_population(population_list, child):
                population_list.append(child)
        population_list = population_sort(population_list)
        population_list = population_list[:pop_size]
        generation += 1
    # plt.show()
    print("finish at", time.time() - start)




if __name__ == "__main__":
    CITIES_NUM = 100
    P_M = 0.1  # probability of mutation
    P_m = 0.03  # gene mutation rate
    c_range = 5 # maximal cities cross
    print("parameter setting:", P_M, P_m, c_range)
    raw_data = np.genfromtxt("cities.csv", encoding="utf-8", delimiter=',', dtype=int)
    dis_matrix = generate_dis_matrix(raw_data)
    adj_list = build_adjcent_list(dis_matrix)

    p_pool = []
    for i in range(1):  # 核数
        p = multiprocessing.Process(target=genetic_algorithm, args=(P_M, P_m, c_range, dis_matrix, adj_list))
        p_pool.append(p)
    for p in p_pool:
        p.start()
    for p in p_pool:
        p.join()
    for p in p_pool:
        p.terminate()
