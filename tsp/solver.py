#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

Point = namedtuple("Point", ['x', 'y'])


def distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))


def get_distances(points, count):
    distances = np.zeros(shape = (2, count, count, 2))
    distances[0, :] = points
    distances[1, :] = points.reshape(count, 1, 2)
    distances = np.sqrt(np.sum((distances[0, :] - distances[1, :]) ** 2, axis = -1))
    """
    distances = []
    for i in tqdm(range(count)):
        j = 0
        d = []
        while j < i + 1:
            d.append(distance(points[i], points[j]))
            j += 1
        distances.append(d)
        """
    return distances


def small_greedy(count, distances):
    number = int(np.log2(count))
    indexes = np.random.choice(count, number, replace=False)
    min_order = greedy(indexes[0], count, distances).copy()
    min_dist = count_path(distances, min_order, count)
    for i in range(1, number):
        order = greedy(indexes[i], count, distances)
        dist = count_path(distances, min_order, count)
        if dist < min_dist:
            min_order = order
    return min_order


def greedy(v0, count, distances):
    order = [v0]
    available = {i for i in range(count)}
    available.remove(order[0])
    for i in tqdm(range(count - 1)):
        min_dist = float("+inf")
        min_vertex = -1
        for vertex in available:
            dist = distances[order[i], vertex]
            if dist < min_dist:
                min_dist = dist
                min_vertex = vertex
        if min_vertex != -1:
            order.append(min_vertex)
            available.remove(min_vertex)

    for vertex in available:
        order.append(vertex)
    return order


def big_greedy(points, count):
    order = [0]
    length = 0
    available = {i for i in range(1, count)}
    for i in tqdm(range(count - 1)):
        min_dist = float("+inf")
        min_vertex = -1
        for vertex in available:
            dist = distance(points[order[i]], points[vertex])
            if dist < min_dist:
                min_dist = dist
                min_vertex = vertex
        if min_vertex != -1:
            length += min_dist
            order.append(min_vertex)
            available.remove(min_vertex)

    for vertex in available:
        order.append(vertex)
    length += distance(points[order[-1]], points[0])
    return order, length


"""
def invert(order, start, end):
    if start > end:
        end, start = start, end
    order[start:end + 1] = order[end:start-1 if start > 0 else -len(order) - 1:-1]
    return order
"""


def invert(order, start, end):
    if start > end:
        end, start = start - 1, end + 1
    rev = reversed(order[start: end + 1])
    order[start: end + 1] = rev


def path_proverka_k(points, order, count, distances):
    max_orders = int(3 * np.log2(count))
    found_solution = False
    prev_orders = []
    while not found_solution:
        length = len(prev_orders)
        if length > 0:
            rand = np.random.randint(0, 20)
            if rand == 0:
                rand_ind = np.random.randint(0, length)
                order = prev_orders[rand_ind]
        patience = 0
        permutation = np.random.permutation(count)
        for i in permutation:
            if length == max_orders:
                rand_ind = np.random.choice(count, 1, p = [max_orders - i for i in range(max_orders)])
                prev_orders[rand_ind] = order.copy()
            else:
                prev_orders.append(order.copy())
            changed, order = k_opt(order.index(i), order, count, distances)
            if not changed:
                patience += 1
        if patience == count:
            found_solution = True
    return order


def k_opt(t1, order, count, distances):
    changed = False

    # сохраняем начальный порядок
    min_order = order.copy()
    t2 = (t1 + 1) % count

    # доступные для рассмотрения вершины
    available = {i for i in range(count)}
    available.remove(order[t1])
    available.remove(order[t2])
    available.remove(order[(t2 + 1) % count])

    # Разница между путями
    difference = 0
    min_diff = 0
    k_max = int(4 * np.sqrt(count))
    for k in range(k_max):
        d1 = d2 = distances[order[t1], order[t2]]
        t3 = -1
        # Ищем ребро (t2, t3) < (t1, t2)
        for ind in range(count):
            if order[ind] in available and ind != t2:
                d = distances[order[ind], order[t2]]
                if d < d2:
                    d2 = d
                    t3 = ind
                    #break
        #else:
        if t3 == -1:
            break

        available.remove(order[t3])
        t4 = (t3 - 1) % count

        difference += d2 + distances[order[t1], order[t4]] - d1 - distances[order[t3], order[t4]]
        invert(order, t2, t4)

        if difference < min_diff:
            changed = True
            min_diff = difference
            min_order = order.copy()
    return changed, min_order


def count_path(distances, order, count):
    obj = distances[order[-1], order[0]]
    for index in range(count - 1):
        obj += distances[order[index], order[index + 1]]
    return obj


def path_proverka(points, order, count, distances):
    found_solution = False
    while not found_solution:
        patience = tqdm(0)
        permutation = np.random.permutation(count)
        for i in permutation:
            if not two_opt(i, order, count, distances):
                patience += 1
                #print(patience)
        if patience == count:
            found_solution = True
    return order


def two_opt(v1, order, count, distances):
    u1 = (v1 + 1) % count
    u2 = -1
    d1 = d2 = distances[order[v1], order[u1]]
    for i in range(count):
        if i not in [v1, u1, (u1 + 1) % count]:
            d = distances[order[u1], order[i]]
            if d < d2:
                d2 = d
                u2 = i

    if d2 < d1:
        v2 = (u2 - 1) % count
        diff = distances[order[v1], order[v2]] - distances[order[v2], order[u2]] - d1 + d2

        if diff < 0:
            invert(order, u1, v2)
            return True
    return False


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')
    nodeCount = int(lines[0])
    points = []
    for i in range(nodeCount):
        line = lines[i+1]
        parts = line.split()
        a = [float(parts[0]), float(parts[1])]
        points.append(a)

    points = np.array(points)

    if nodeCount < 10000:
        distances = get_distances(points, nodeCount)
        order = small_greedy(nodeCount, distances)

        solution = path_proverka_k(points, order, nodeCount, distances)
        obj = count_path(distances, solution, nodeCount)
    # calculate the length of the tour
    else:
        solution, obj = big_greedy(points, nodeCount)

    plt.figure(figsize=(5, 5))
    x = points[solution, 0]
    x = np.append(x, points[solution[0], 0])
    y = points[solution, 1]
    y = np.append(y, points[solution[0], 1])
    plt.title("After")
    plt.plot(x, y)
    plt.show()

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

"""
def main():
    solve_it("51\n\
27 68\n\
30 48\n\
43 67\n\
58 48\n\
58 27\n\
37 69\n\
38 46\n\
46 10\n\
61 33\n\
62 63\n\
63 69\n\
32 22\n\
45 35\n\
59 15\n\
5 6\n\
10 17\n\
21 10\n\
5 64\n\
30 15\n\
39 10\n\
32 39\n\
25 32\n\
25 55\n\
48 28\n\
56 37\n\
30 40\n\
37 52\n\
49 49\n\
52 64\n\
20 26\n\
40 30\n\
21 47\n\
17 63\n\
31 62\n\
52 33\n\
51 21\n\
42 41\n\
31 32\n\
5 25\n\
12 42\n\
36 16\n\
52 41\n\
27 23\n\
17 33\n\
13 13\n\
57 58\n\
62 42\n\
42 57\n\
16 57\n\
8 52\n\
7 38\n\
")

main()
"""
