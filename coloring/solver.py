#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import defaultdict


class Graph:
    def __init__(self, n, m, nodes):
        self.n = n
        self.m = m
        self.nodes = nodes

    def graph_proverka(self, node, color):
        nodes = sorted(self.nodes, key=lambda v: v.name)
        for n in node.neighbours:
            if nodes[n].color == color:
                return False
        for n in node.neighbours:
            nodes[n].degree -= 1
        return True

    def get_solution(self):
        nodes = sorted(self.nodes, key = lambda v : v.name)
        solution = [v.color for v in nodes]
        return max(solution) + 1, solution


class Node:
    def __init__(self, name, neighbours):
        self.name = name
        self.neighbours = neighbours
        self.degree = len(neighbours)
        self.color = -1

    def __str__(self):
        return f"Я вершина {self.name} степени {self.degree}, это мои соседи: {self.neighbours}, мой цвет: {self.color}"


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])
    #print(node_count, edge_count)

    edges = defaultdict(set)
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        #print(parts)
        edges[int(parts[0])].add(int(parts[1]))
        edges[int(parts[1])].add(int(parts[0]))

    # build a trivial solution
    # every node has its own color

    '''
    solution = range(0, node_count)
    '''

    # From the internet
    nodes = []
    for vertex in edges:
        nodes.append(Node(vertex, edges[vertex]))

    graph = Graph(node_count, edge_count, nodes)

    sorted_nodes = sorted(nodes, key=lambda v : v.degree, reverse=True)

    """print("sorted")
    for node in sorted_nodes:
        print(node)"""

    color = 0
    num_change = 1
    while num_change > 0:
        num_change = 0
        for node in sorted_nodes:
            if node.color == -1:
                if graph.graph_proverka(node, color):
                    node.color = color
                    #print(node.name, "покрашен в", color)
                    num_change += 1
        color += 1
        sorted_nodes = sorted(nodes, key=lambda v: v.degree, reverse=True)

    solution = graph.get_solution()

    # prepare the solution in the specified output format
    output_data = str(solution[0]) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution[1]))

    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

