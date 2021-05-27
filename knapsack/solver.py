#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from collections import deque

from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])


def estimate(sorted_items, ones, zeros, cap):
    value = 0
    for i in range(len(sorted_items)):
        if (i not in ones) and (i not in zeros):
            item = sorted_items[i]
            if item.weight <= cap:
                cap -= item.weight
                value += item.value
            else:
                value += item.value / item.weight * cap
                break
    return value


class Node:
    def __init__(self, value, weight, sorted_items, ones, zeros, i, cap):
        self.value = value
        self.weight = weight
        self.estimate = value + estimate(sorted_items, ones, zeros, cap)
        self.ones = ones
        self.zeros = zeros
        self.i = i
        self.cap = cap
        self.leaf = i == len(sorted_items)

    def __str__(self):
        return f"Я узел с такой ценностью: {self.value} и такой оценкой: {self.estimate}, вешу {self.weight}кг, это я беру: {self.ones}, а это я не беру: {self.zeros}\n" \
               f"осталось места: {self.cap}"


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i, int(parts[0]), int(parts[1])))

    # a trivial algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full
    value = 0
    weight = 0
    taken = [0]*len(items)

    # Very simple solution: 3/10 on every

    '''
    for item in items:
        if weight + item.weight <= capacity:
            taken[item.index - 1] = 1
            value += item.value
            weight += item.weight
    '''

    # Dynamic programming: 10/10 on 1,2,3,5

    '''
    if item_count * capacity > 5e8:
        opt = 0
        sorted_items = sorted(items, key=lambda item: item.value / item.weight, reverse=True)
        remaining_weight = capacity
        for item in sorted_items:
            if item.weight <= remaining_weight:
                taken[item.index - 1] = 1
                value += item.value
                remaining_weight -= item.weight
            if remaining_weight == 0:
                break

    else:
        opt = 1
        dynamic_table = np.zeros((capacity + 1, item_count + 1))

        for item in items:
            if item.weight <= capacity:
                dynamic_table[item.weight:, item.index] = np.maximum(dynamic_table[item.weight:, item.index - 1],
                                                                     item.value + dynamic_table[:-item.weight,
                                                                                  item.index - 1])
                dynamic_table[:item.weight, item.index] = dynamic_table[:item.weight, item.index - 1]
            else:
                dynamic_table[:, item.index] = dynamic_table[:, item.index - 1]

        current_ind = capacity
        value = int(dynamic_table[-1, -1])
        for i in range(item_count, 0, -1):
            if dynamic_table[current_ind, i] != dynamic_table[current_ind, i - 1]:
                current_ind -= items[i - 1].weight
                taken[i - 1] = 1
    '''

    # Greedy value density

    '''
    sorted_items = sorted(items, key=lambda item: item.value / item.weight, reverse=True)
    remaining_weight = capacity
    for item in sorted_items:
        if item.weight <= remaining_weight:
            taken[item.index - 1] = 1
            value += item.value
            remaining_weight -= item.weight
        if remaining_weight == 0:
            break
    '''

    # Branch and bound 10/10!!

    sorted_items = sorted(items, key = lambda item: item.value / item.weight, reverse=True)
    max_estimate = 0
    ones = set()
    zeros = set()
    root = Node(0, 0, sorted_items, ones, zeros, 0, capacity)
    best_node = root
    stack = deque()
    stack.append(root)
    while len(stack) != 0:
        node = stack.pop()
        zeros_for_zero = node.zeros.copy()
        ones_for_one = node.ones.copy()
        zeros_for_zero.add(node.i)
        zero = Node(node.value, node.weight, sorted_items, node.ones, zeros_for_zero, node.i + 1, node.cap)
        if zero.estimate >= max_estimate:
            if zero.leaf:
                max_estimate = zero.value
                best_node = zero
            else:
                stack.append(zero)

        new_value = node.value + sorted_items[node.i].value
        if sorted_items[node.i].weight <= node.cap:
            ones_for_one.add(node.i)
            one = Node(new_value, node.weight + sorted_items[node.i].weight, sorted_items, ones_for_one, node.zeros,
                       node.i + 1, node.cap - sorted_items[node.i].weight)
            if one.estimate >= max_estimate:
                if one.leaf:
                    max_estimate = one.value
                    best_node = one
                else:
                    stack.append(one)

    #print(best_node)
    opt = 1
    value = best_node.value
    for num in best_node.ones:
        taken[sorted_items[num].index - 1] = 1

    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(opt) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')