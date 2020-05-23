#!/usr/bin/python

from collections import defaultdict
import csv 

class Graph():
    def __init__(self):
        """
        self.edges is a dict of all possible next nodes
        e.g. {'X': ['A', 'B', 'C', 'E'], ...}
        self.weights has all the weights between two nodes,
        with the two nodes as a tuple as the key
        e.g. {('X', 'A'): 7, ('X', 'B'): 2, ...}
        """
        self.edges = defaultdict(list)
        self.weights = {}
        self.nodes_set = set()
        self.nodes_list = []

    
    def add_edge(self, from_node, to_node, weight):
        # Note: assumes edges are bi-directional
        if from_node not in self.nodes_set:
            self.nodes_set.add(from_node)
            self.nodes_list.append(from_node)

        if to_node not in self.nodes_set:
            self.nodes_set.add(to_node)
            self.nodes_list.append(to_node)

        self.edges[from_node].append(to_node)
        # self.edges[to_node].append(from_node)
        self.weights[(from_node, to_node)] = weight
        # self.weights[(to_node, from_node)] = weight

    def create_csv(self):
        with open('../../matlab/gr_optimal_control_3rd_order_int.csv', 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            for nodes, weight in self.weights.items():
                writer.writerow([self.nodes_list.index(nodes[0])+1, self.nodes_list.index(nodes[1])+1, int(weight*1000000)])

def dijkstra(graph, initial, end):
    # shortest paths is a dict of nodes
    # whose value is a tuple of (previous node, weight)
    shortest_paths = {initial: (None, 0)}
    current_node = initial
    visited = set()
    
    while current_node != end:
        visited.add(current_node)
        destinations = graph.edges[current_node]
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node in destinations:
            weight = graph.weights[(current_node, next_node)] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)
        
        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            return "Route Not Possible"
        # next node is the destination with the lowest weight
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])
    
    # Work back through destinations in shortest path
    path = []
    path_idxs = []
    while current_node is not None:
        path.append(current_node)
        path_idxs.append(graph.nodes_list.index(current_node))
        next_node = shortest_paths[current_node][0]
        current_node = next_node
    # Reverse path
    path = path[::-1]
    path_idxs = path_idxs[::-1]
    print("initial (index): ", initial, graph.nodes_list.index(initial))
    print("end (index): ", end, graph.nodes_list.index(end))
    print("dijkstra path: ", path)
    print("dijkstra path (indexes): ", path_idxs)
    return path
