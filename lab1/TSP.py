import numpy as np
import copy
import time
import sys
import random
"""
solution representation list
    this could be cheep where apply neighbor changing
"""

def evaluateTSP(s):
    global evaluation_num
    evaluation_num+=1
    result = 0
    for i in range(len(s)):
        c1 = s[i - 1]
        c2 = s[i]
        result += dis_m[c1][c2]
    return result


def evaluateFS(s):
    global evaluation_num
    evaluation_num+=1
    machines=np.zeros(mc_num+1)
    for i in s:
        for j in range(len(machines)-1):
            machines[j]=max(machines[j-1], machines[j])+wk_time[i][j]
    return machines[-2]

class Solution:
    def __init__(self, new_result, cost=None):
        self.result=new_result
        self.evaluation=cost
        if not cost:
            self.evaluate()
    def evaluate(self):
        self.evaluation=evaluate(self.result)
    def __len__(self):
        return len(self.result)
    def __setitem__(self, key, value):
        self.result[key]=value
    def __getitem__(self, item):
        return self.result[item]
    def __lt__(self, other):
        return self.evaluation < other.evaluation
    def __gt__(self, other):
        return self.evaluation > other.evaluation
    def __eq__(self, other):
        return self.evaluation == other.evaluation
    def __ge__(self, other):
        return self.evaluation >= other.evaluation
    def __le__(self, other):
        return self.evaluation <= other.evaluation



"""----------neighbor-------------"""


def inversionBest(s:Solution):
    original_cost=s.evaluation
    best_cost=s.evaluation
    best=s
    ss=copy.deepcopy(s)
    for i in range(problem_size):
        for j in range(problem_size-i):
            solution = copy.deepcopy(ss)
            tmp=solution.result[0:j]
            tmp.reverse()
            solution.result[0:j]=tmp
            solution.evaluate()
            if solution.evaluation < best_cost:
                best_cost=solution.evaluation
                best=solution
            else:
                tmp = solution.result[0:j]
                tmp.reverse()
                solution.result[0:j] = tmp
                solution.evaluation=best_cost
        v=ss.result.pop(0)
        ss.result.append(v)
        ss.evaluate()
    if original_cost <= best.evaluation:
        return None
    else:
        return best

def inversionfirst(s:Solution):
    original_cost=s.evaluation
    best_cost=s.evaluation
    best=s
    ss=copy.deepcopy(s)
    solution=None
    for i in range(problem_size):
        for j in range(problem_size-i):
            solution = copy.deepcopy(ss)
            tmp=solution.result[0:j]
            tmp.reverse()
            solution.result[0:j]=tmp
            solution.evaluate()
            if solution.evaluation < best_cost:
                break
        if solution.evaluation < best_cost:
            break
        v=ss.result.pop(0)
        ss.result.append(v)
        ss.evaluate()
    if original_cost <= solution.evaluation:
        return None
    else:
        return solution



def adj2ExchangeBestMove(solution:Solution):
    """
    1. adjacent exchange 2 gene
    2. best move strategy
    3. shadow copy
    :param solution:
    :return: if no better return None
    """
    original_cost=solution.evaluation
    best_cost=solution.evaluation
    best=solution
    for i in range(len(solution)):
        solution[i-1], solution[i]=solution[i], solution[i-1]
        solution.evaluate()
        if solution.evaluation < best_cost:
            best_cost=solution.evaluation
            best=copy.deepcopy(best)
        solution[i - 1], solution[i] = solution[i], solution[i - 1]
    if original_cost <= best.evaluation:
        return None
    else:
        return best


def adj2ExchangeFirstMove(solution:Solution):
    """
    1. adjacent exchange 2 gene
    2. first move strategy
    3. shadow copy
    :param solution:
    :return: if no better return None
    """
    original_cost=solution.evaluation
    for i in range(len(solution)):
        solution[i-1], solution[i]=solution[i], solution[i-1]
        solution.evaluate()
        if solution.evaluation < original_cost:
            break
        else:
            solution[i - 1], solution[i] = solution[i], solution[i - 1]
    if original_cost <= solution.evaluation:
        return None
    else:
        return solution


def insertBestSingle(solution:Solution, index=None):

    """
    1. insert a city to other place
    2. best move strategy
    :param solution:
    :return:
    """
    original_cost=solution.evaluation
    best_cost=solution.evaluation
    l=copy.deepcopy(solution.result)  # in order to speed up
    lenl=len(l)
    cost=solution.evaluation
    best=l
    for j in range(lenl):
        value=l.pop(0)
        for i in range(lenl):
            l.insert(i, value)
            cost = evaluate(l)
            if cost < best_cost:
                best_cost=cost
                best=copy.deepcopy(l)
            else:
                l.pop(i)
        l.append(value)
    if original_cost <= best_cost:
        return None
    else:
        return Solution(best, best_cost)


def insertFirstSingle(solution:Solution):

    """
    1. insert a city to other place
    2. first move strategy
    3. shadow copy
    :param solution:
    :return:
    """
    original_cost=solution.evaluation
    best_cost=solution.evaluation
    l=copy.deepcopy(solution.result)  # in order to speed up
    lenl=len(l)
    cost=solution.evaluation
    for j in range(lenl):
        value=l.pop(0)
        for i in range(lenl):
            l.insert(i, value)
            cost = evaluate(l)
            if cost < best_cost:
                break
            else:
                l.pop(i)
        if cost < best_cost:
                break
        l.append(value)
    if original_cost <= cost:
        return None
    else:
        return Solution(l, cost)


def insertRandomSingle(solution:Solution):

    """
    1. insert a city to other place
    2. first move strategy
    3. shadow copy
    :param solution:
    :return:
    """
    original_cost=solution.evaluation
    r1, r2=np.random.randint(problem_size), np.random.randint(problem_size)
    if r1==r2:
        r1-=1
    l=list(solution.result)
    value=l.pop(r1)
    l.insert(min(r2, len(l)), value)
    cost=evaluate(l)
    if original_cost <= cost:
        return None
    else:
        return Solution(l, cost)


def arbTwoFirstExchange(solution:Solution):
    original_cost=solution.evaluation
    best_cost=solution.evaluation
    best=solution
    for i in range(problem_size):
        for j in range(i, problem_size):
            solution[i], solution[j] = solution[j], solution[i]
            solution.evaluate()
            if solution.evaluation < best_cost:
                best_cost=solution.evaluation
                best=copy.deepcopy(solution)
                break
            else:
                solution[i], solution[j] = solution[j], solution[i]
    if original_cost <= best.evaluation:
        return None
    else:
        return best


def arbTwoBestExchange(solution:Solution):
    original_cost=solution.evaluation
    best_cost=solution.evaluation
    best=solution
    for i in range(problem_size):
        for j in range(i, problem_size):
            solution[i], solution[j] = solution[j], solution[i]
            solution.evaluate()
            if solution.evaluation < best_cost:
                best_cost=solution.evaluation
                best=copy.deepcopy(solution)
            else:
                solution[i], solution[j] = solution[j], solution[i]
    if original_cost <= best.evaluation:
        return None
    else:
        return best

def selfing(parent:Solution):
    point1, point2 = random.sample(range(problem_size), 2)  # check
    if point1>point2:
        point1, point2=point2, point1
    tmp1 = parent[:point1]
    tmp2 = parent[point1:point2]
    tmp3 = parent[point2:]
    tmp = tmp1+tmp3
    point = np.random.randint(len(tmp))
    tmp1 = tmp[:point]
    tmp3 = tmp[point:]
    child = tmp1+tmp2+tmp3
    # assert  len(child)==job_num
    new_individual = Solution(child)
    # print(new_individual.evaluation, new_individual.result)
    if new_individual.evaluation < parent.evaluation:
        return new_individual
    else:
        return None
    # l=list(solution.result)
    # b=np.random.randint(problem_size)
    # c=np.random.randint(size)
    # fa, fb, fc=copy.deepcopy(l[len(l)-b-c:b]), \
    #            copy.deepcopy(l[b:(b+c)%len(l)]), \
    #            copy.deepcopy(l[(b+c)%len(l):len(l)-(b+c)])
    # fac=fa+fc
    # insert_index=np.random.randint(len(fac))
    # fa, fc=fac[0:insert_index], fac[insert_index:]
    # f=[fa+fb+fc, fa+fb.reverse()+fc]
    # cost=(solution.evaluation, evaluate(f[0]), evaluate(f[1]))
    # min_which = np.argmin(cost)
    # if min_which == 0:
    #     return None
    # else:
    #     return Solution(f[min_which+1], min(cost))



def insertFirstMany(solution:Solution, size):
    """
    1. insert a seriel of cities to other place
    2. first move strategy
    3. shadow copy
    :param solution:
    :return:
    """
    # original_cost = solution.evaluation
    # divide into 3 part
    # a b c
    l=list(solution.result)
    b=np.random.randint(problem_size)
    c=np.random.randint(size)
    fa, fb, fc=copy.deepcopy(l[len(l)-b-c:b]), \
               copy.deepcopy(l[b:(b+c)%len(l)]), \
               copy.deepcopy(l[(b+c)%len(l):len(l)-(b+c)])
    fac=fa+fc
    insert_index=np.random.randint(len(fac))
    fa, fc=fac[0:insert_index], fac[insert_index:]
    f=[fa+fb+fc, fa+fb.reverse()+fc]
    cost=(solution.evaluation, evaluate(f[0]), evaluate(f[1]))
    min_which = np.argmin(cost)
    if min_which == 0:
        return None
    else:
        return Solution(f[min_which+1], min(cost))


def __insertFirstMany_size__():
    return int(problem_size/10e6*evaluation_num)



"""----------neighbor-------------"""


"""----------TSP basic preprocessing ---------------------"""

def build_adjcent_list(distance_matrix):
    adjcent_dict_list = dict()
    for i in range(problem_size):
        adjcent_dict_list[i] = sorted(list(range(problem_size)), key=lambda index:distance_matrix[i][index])[1:]
        # remove itself
    return adjcent_dict_list


def cal_distance(c1, c2):
    tmp = c1 - c2
    return (tmp[0] * tmp[0] + tmp[1] * tmp[1]) ** 0.5


def generate_dis_matrix(loc_raw_data):
    """

    :return:
    """
    dis = np.zeros(shape=(problem_size, problem_size))
    for i in range(problem_size):
        for j in range(i+1, problem_size):
            dis[i][j] = cal_distance(loc_raw_data[i], loc_raw_data[j])
            dis[j][i] = dis[i][j]
    return dis


"""----------TSP basic preprocessing ---------------------"""

"""----------path scanning greedy--------------------------------"""
def greedy_init(start):
    visit_node = set()
    visit_node.add(start)
    ordered_list = [start]
    while len(ordered_list) < problem_size:
        for aj_node in adj_list[start]:
            if aj_node not in visit_node:
                visit_node.add(aj_node)
                ordered_list.append(aj_node)
                start = aj_node
                break
    return ordered_list
"""----------path scanning greedy--------------------------"""


if __name__ == "__main__":
    begin=time.time()
    problem="FS"
    evaluation_num = 0
    raw_data = np.genfromtxt("cities.csv", encoding="utf-8", delimiter=',', dtype=int)
    problem_size=raw_data.shape[0]
    evaluate=None
    dis_matrix=None
    adj_list=None

    mc_num=5
    job_num=20
    wk_time=None
    if problem=="TSP":
        dis_m = generate_dis_matrix(raw_data)
        adj_list = build_adjcent_list(dis_m)
        evaluate=evaluateTSP ##
        neighbor_f = inversionBest ##
        best_cost=sys.maxsize
        best=None
        pool=[]
        for i in range(problem_size):
            # s=greedy_init(i)
            # cost=evaluate(s)
            s = random.sample(range(100), 100)
            cost= evaluate(s)
            pool.append(Solution(s, cost))
        i=0
        pool=sorted(pool)
        print(evaluation_num, ',', pool[0].evaluation)
        while evaluation_num<1e6:
            tmps=neighbor_f(copy.deepcopy(pool[i]))
            if tmps:
                pool[i]=tmps
                print(evaluation_num, ',', tmps.evaluation)
            else:
                i=(i+1)%problem_size
                print(evaluation_num, ',', pool[i].evaluation)
    elif problem=="FS":
        problem_size = job_num
        evaluate=evaluateFS ##
        print('inversionfirst')
        neighbor_f = inversionfirst ##
        # wk_time=np.array(10+90*np.random.rand(job_num, mc_num), dtype=int)
        wk_time=np.array([[44, 21, 54, 62, 52],
 [40, 12, 22, 75, 74],
 [74, 59, 89, 75, 95],
 [59, 38, 17, 24, 52],
 [73, 93, 78, 60, 84],
 [97, 75, 13, 27, 32],
 [92, 62, 28, 19, 48],
 [23, 34, 91, 54, 45],
 [33, 74, 38, 82, 66],
 [66, 47, 74, 11, 91],
 [92, 63, 61, 13, 81],
 [12, 96, 10, 59, 14],
 [67, 98, 20, 54, 71],
 [62, 16, 20, 88, 20],
 [15, 33, 43, 51, 63],
 [14, 96, 99, 31, 91],
 [77, 46, 98, 25, 34],
 [88, 83, 32, 43, 94],
 [70, 11, 86, 38, 82],
 [73, 86, 34, 68, 80]],dtype=int)
        s = random.sample(range(job_num), job_num)
        cost= evaluate(s)
        s=Solution(s, cost)
        print(evaluation_num, ',', s.evaluation)
        while evaluation_num<1e5:
            tmps=neighbor_f(copy.deepcopy(s))
            if tmps:
                s=tmps
                print(evaluation_num, ',', tmps.evaluation)
            else:
                s = random.sample(range(job_num), job_num)
                cost = evaluate(s)
                s = Solution(s, cost)
                print(evaluation_num, ',', s.evaluation)
    print(time.time()-begin)