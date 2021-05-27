#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])

def distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))


def get_distances(points, count):
    distances = np.zeros(shape = (2, count, count, 2))
    distances[0, :] = points
    distances[1, :] = points.reshape(count, 1, 2)
    distances = np.sqrt(np.sum((distances[0, :] - distances[1, :]) ** 2, axis = -1))
    return distances


def get_data(points, count, demands, cap, num_vehicles):
    data = {}
    data["distance_matrix"] = get_distances(points, count)
    data["demands"] = demands
    data['vehicle_capacities'] = [cap] * num_vehicles
    data['num_vehicles'] = num_vehicles
    data['depot'] = 0
    return data


def length(customer1, customer2):
    return math.sqrt((customer1.x - customer2.x)**2 + (customer1.y - customer2.y)**2)


def solve_it(input_data):
    # Modify this code to run your optimization algorithm
    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    customer_count = int(parts[0])
    vehicle_count = int(parts[1])
    vehicle_capacity = int(parts[2])

    customers = []
    demands = []
    points = []
    for i in range(1, customer_count+1):
        parts = lines[i].split()
        demands.append(int(parts[0]))
        points.append([float(parts[1]), float(parts[2])])
        customers.append(Customer(i-1, int(parts[0]), float(parts[1]), float(parts[2])))
    points = np.array(points)

    data = get_data(points, customer_count, demands, vehicle_capacity, vehicle_count)
    manager = pywrapcp.RoutingIndexManager(data['distance_matrix'].shape[0],
                                           data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = 300
    solution = routing.SolveWithParameters(search_parameters)

    output = ''
    if not solution:
        depot = customers[0]
        vehicle_tours = []

        remaining_customers = set(customers)
        remaining_customers.remove(depot)

        for v in range(0, vehicle_count):
            # print "Start Vehicle: ",
            vehicle_tours.append([])
            capacity_remaining = vehicle_capacity
            while sum([capacity_remaining >= customer.demand for customer in remaining_customers]) > 0:
                used = set()
                order = sorted(remaining_customers,
                               key=lambda customer: -customer.demand * customer_count + customer.index)
                for customer in order:
                    if capacity_remaining >= customer.demand:
                        capacity_remaining -= customer.demand
                        vehicle_tours[v].append(customer)
                        # print '   add', ci, capacity_remaining
                        used.add(customer)
                remaining_customers -= used
        obj = 0
        for v in range(0, vehicle_count):
            vehicle_tour = vehicle_tours[v]
            if len(vehicle_tour) > 0:
                obj += length(depot, vehicle_tour[0])
                for i in range(0, len(vehicle_tour) - 1):
                    obj += length(vehicle_tour[i], vehicle_tour[i + 1])
                obj += length(vehicle_tour[-1], depot)
        for v in range(0, vehicle_count):
            output += str(depot.index) + ' ' + ' '.join(
                [str(customer.index) for customer in vehicle_tours[v]]) + ' ' + str(depot.index) + '\n'
    else:
        obj = solution.ObjectiveValue()
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                output += '{0} '.format(node_index)
                index = solution.Value(routing.NextVar(index))
            output += '{0}\n'.format(manager.IndexToNode(index))

    # prepare the solution in the specified output format
    outputData = '%.2f' % obj + ' ' + str(0) + '\n'

    return outputData + output


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:

        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)')

