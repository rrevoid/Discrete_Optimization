#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple, defaultdict
import math
from time import time
import numpy as np

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])


def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)


def change_facility(solution, customers, facilities, capacity_remaining, customers_in_facilities):
    prev_solution = solution.copy()
    rand_ind = np.random.permutation(len(customers))
    for ind in rand_ind:
        customer = customers[ind]
        for facility in facilities:
            curr_ind = solution[customer.index]
            curr_facility = facilities[curr_ind]
            if facility.index != curr_ind and capacity_remaining[facility.index] >= customer.demand:
                prev_cost = length(customer.location, curr_facility.location)
                curr_cost = length(customer.location, facility.location) -\
                            (len(customers_in_facilities[curr_ind]) == 1) * curr_facility.setup_cost + \
                            (len(customers_in_facilities[facility.index]) == 0) * facility.setup_cost
                if curr_cost < prev_cost:
                    customers_in_facilities[curr_ind].discard(customer)
                    customers_in_facilities[facility.index].add(customer)
                    capacity_remaining[curr_ind] = actual_rem_cap(customers_in_facilities, curr_facility)
                    capacity_remaining[facility.index] = actual_rem_cap(customers_in_facilities, facility)
                    solution[customer.index] = facility.index
    return prev_solution != solution


def swap_two_customers(solution, facilities, customers, capacity_remaining, customers_in_facilities):
    prev_solution = solution.copy()
    rand_ind = np.random.permutation(len(customers))
    for ind in rand_ind:
        customer1 = customers[ind]
        for customer2 in customers:
            ok = False
            facility1 = solution[customer1.index]
            facility2 = solution[customer2.index]

            if facility1 > facility2:
                cap1 = capacity_remaining[facility1] + customer1.demand - customer2.demand
                cap2 = capacity_remaining[facility2] + customer2.demand - customer1.demand
                if cap1 >= 0 and cap2 >= 0:
                    prev_cost = length(customer1.location, facilities[facility1].location) +\
                                length(customer2.location, facilities[facility2].location)
                    curr_cost = length(customer1.location, facilities[facility2].location) +\
                                length(customer2.location, facilities[facility1].location)
                    if curr_cost < prev_cost:
                        customers_in_facilities[facility1].discard(customer1)
                        customers_in_facilities[facility2].discard(customer2)
                        customers_in_facilities[facility1].add(customer2)
                        customers_in_facilities[facility2].add(customer1)
                        capacity_remaining[facility1] = actual_rem_cap(customers_in_facilities, facilities[facility1])
                        capacity_remaining[facility2] = actual_rem_cap(customers_in_facilities, facilities[facility2])

                        solution[customer1.index] = facility2
                        solution[customer2.index] = facility1
    return prev_solution != solution


def sum_length(facility, customers):
    l = 0
    for customer in customers:
        l += length(customer.location, facility.location)
    return l


def actual_rem_cap(customers_in_facilities, facility):
    occ = 0
    for cust in customers_in_facilities[facility.index]:
        occ += cust.demand
    return facility.capacity - occ


def swap_two_facilities(solution, facilities, customers_in_facilities, capacity_remaining, customer_count):
    prev_solution = solution.copy()
    rand_ind = np.random.permutation(len(facilities))
    for ind in rand_ind:
        facility1 = facilities[ind]
        for facility2 in facilities:
            if len(customers_in_facilities[facility1.index]) == 0 or len(customers_in_facilities[facility2.index]) == 0:
                if len(customers_in_facilities[facility1.index]) != 0:
                    if facility2.capacity >= facility1.capacity - capacity_remaining[facility1.index]:
                        prev_cost = sum_length(facility1, customers_in_facilities[facility1.index])
                        curr_cost = sum_length(facility2, customers_in_facilities[facility1.index]) - \
                                    facility1.setup_cost + facility2.setup_cost
                        if curr_cost < prev_cost:
                            customers_in_facilities[facility1.index], customers_in_facilities[facility2.index] = \
                                customers_in_facilities[facility2.index], customers_in_facilities[facility1.index]

                            capacity_remaining[facility1.index] = actual_rem_cap(customers_in_facilities, facility1)
                            capacity_remaining[facility2.index] = actual_rem_cap(customers_in_facilities, facility2)
                            for cust in range(customer_count):
                                if solution[cust] == facility1.index:
                                    solution[cust] = facility2.index
            else:
                if facility1.index > facility2.index:
                    occ_1 = facility1.capacity - capacity_remaining[facility1.index]
                    occ_2 = facility2.capacity - capacity_remaining[facility2.index]
                    if occ_1 <= facility2.capacity and occ_2 <= facility1.capacity:
                        prev_cost = sum_length(facility1, customers_in_facilities[facility1.index]) + \
                                    sum_length(facility2, customers_in_facilities[facility2.index])
                        curr_cost = sum_length(facility1, customers_in_facilities[facility2.index]) + \
                                    sum_length(facility2, customers_in_facilities[facility1.index])
                        if curr_cost < prev_cost:
                            customers_in_facilities[facility1.index], customers_in_facilities[facility2.index] =\
                                customers_in_facilities[facility2.index], customers_in_facilities[facility1.index]
                            capacity_remaining[facility1.index] = facility1.capacity - occ_2 #actual_rem_cap(customers_in_facilities, facility1)
                            capacity_remaining[facility2.index] = facility2.capacity - occ_1 #actual_rem_cap(customers_in_facilities, facility2)
                            for cust in range(customer_count):
                                if solution[cust] == facility1.index:
                                    solution[cust] = facility2.index
                                elif solution[cust] == facility2.index:
                                    solution[cust] = facility1.index
    return prev_solution != solution


def add_new(prev_orders, solution, max_orders, len_orders, capacity_remaining, prev_caps):
    if len_orders < max_orders:
        prev_orders.append(solution.copy())
        prev_caps.append(capacity_remaining.copy())
        len_orders += 1
    else:
        ps = np.array([i for i in range(1, len_orders + 1)])
        rand = np.random.choice(len_orders, 1, p = ps/ps.sum())
        prev_orders[rand[0]] = solution.copy()
        prev_caps[rand[0]] = capacity_remaining.copy()


def local_search(greedy_solution, customers, facilities, facility_count, customer_count, capacity_remaining, tl = 600):
    solution = greedy_solution
    customers_in_facilities = defaultdict(set)
    max_orders = int(3 * np.log2(facility_count))
    len_orders = 0
    prev_orders = [solution.copy()]
    prev_caps = [capacity_remaining.copy()]

    for customer in range(customer_count):
        customers_in_facilities[solution[customer]].add(customers[customer])

    found_solution = False
    start_time = time()
    while not found_solution:
        rand = np.random.randint(0, 20)
        if rand == 1 and len_orders > 0:
            rand_ind = np.random.randint(0, len_orders)
            solution = prev_orders[rand_ind]
            capacity_remaining = prev_caps[rand_ind]

        changed1 = swap_two_facilities(solution, facilities, customers_in_facilities, capacity_remaining, customer_count)
        if changed1:
            add_new(prev_orders, solution, max_orders, len_orders, capacity_remaining, prev_caps)
        changed2 = swap_two_customers(solution, facilities, customers, capacity_remaining, customers_in_facilities)
        if changed2:
            add_new(prev_orders, solution, max_orders, len_orders, capacity_remaining, prev_caps)
        changed3 = change_facility(solution, customers, facilities, capacity_remaining, customers_in_facilities)
        if changed3:
            add_new(prev_orders, solution, max_orders, len_orders, capacity_remaining, prev_caps)
        if not (changed1 or changed2 or changed3):
            found_solution = True
        curr_time = time()
        if (curr_time - start_time) > tl:
            found_solution = True
    used = [0] * facility_count
    for cust in range(customer_count):
        used[solution[cust]] = 1
    return solution, used


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input

    lines = input_data.split('\n')

    parts = lines[0].split()

    facility_count = int(parts[0])
    customer_count = int(parts[1])

    facilities = []
    for i in range(1, facility_count + 1):
        parts = lines[i].split()
        facilities.append(Facility(i - 1, float(parts[0]), int(parts[1]), Point(float(parts[2]), float(parts[3]))))

    customers = []
    for i in range(facility_count + 1, facility_count + 1 + customer_count):
        parts = lines[i].split()
        customers.append(Customer(i - 1 - facility_count, int(parts[0]), Point(float(parts[1]), float(parts[2]))))

    # build a solution
    solution = [-1] * len(customers)
    capacity_remaining = [f.capacity for f in facilities]
    facility_index = 0
    for customer in customers:
        if capacity_remaining[facility_index] >= customer.demand:
            solution[customer.index] = facility_index
            capacity_remaining[facility_index] -= customer.demand
        else:
            facility_index += 1
            assert capacity_remaining[facility_index] >= customer.demand
            solution[customer.index] = facility_index
            capacity_remaining[facility_index] -= customer.demand

    used = [0] * len(facilities)
    for facility_index in solution:
        used[facility_index] = 1

    solution, used_facilities = local_search(solution, customers, facilities, facility_count, customer_count, capacity_remaining)

    obj = sum([f.setup_cost * used_facilities[f.index] for f in facilities])
    for customer in customers:
        obj += length(customer.location, facilities[solution[customer.index]].location)

    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

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
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)')
