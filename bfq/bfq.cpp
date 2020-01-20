// A C++ program for Bellman-Ford's queue-based single source 
// shortest path algorithm. 
#include <bits/stdc++.h>
#include <chrono>
#include <iostream>

using namespace std;
using namespace std::chrono;

typedef pair<int, int> iPair; 
typedef pair<list<iPair>, int> liPair; 

// A utility function used to print the solution 
void printArr(int dist[], int n) 
{ 
	printf("Vertex Distance from Source\n"); 
	for (int i = 0; i < n; ++i) 
		printf("%d \t\t %d\n", i, dist[i]); 
} 

// This class represents a directed graph using 
// adjacency list representation 
class Graph 
{ 
public:
	Graph();

	void addEdge(int u, int v, int w);
	
	vector<list<iPair> > nodes; 
}; 

Graph::Graph() 
{ 
} 

void Graph::addEdge(int u, int v, int w)
{ 
	if (u >= nodes.size())
	{
		nodes.resize(u+1);
	}
	if (v >= nodes.size())
	{
		nodes.resize(v+1);
	}

	nodes[u].push_back(make_pair(v, w)); 
	nodes[v].push_back(make_pair(u, w)); 
} 

// The main function that finds shortest distances from src to 
// all other vertices using Bellman-Ford algorithm.
void BellmanFord(shared_ptr<Graph> graph, int src) 
{ 
	int dist[graph->nodes.size()]; 

	// Step 1: Initialize distances from src to all other vertices 
	// as INFINITE 
	for (int i = 0; i < graph->nodes.size(); i++) 
		dist[i] = INT_MAX;
	dist[src] = 0;

	queue<liPair> node_queue;
	node_queue.push(make_pair(graph->nodes[src], src));
	
	set<int> node_set;
	node_set.insert(src);

	int u;
	int v;
	int weight;
	list<iPair> node;
	list<iPair>::iterator i;
	liPair node_u_pair;

	// Step 2: Relax all edges |V| - 1 times. A simple shortest 
	// path from src to any other vertex can have at-most |V| - 1 
	// edges
	auto start = high_resolution_clock::now(); 
	while(!node_queue.empty())
	{
		node_u_pair = node_queue.front();
		node = node_u_pair.first;
		u = node_u_pair.second;
		node_queue.pop();

		for (i = node.begin(); i != node.end(); ++i)
		{
			v = (*i).first;
			weight = (*i).second;
			if ((dist[u] != INT_MAX) && (dist[u] + weight < dist[v])) 
			{
				dist[v] = dist[u] + weight;
				if (node_set.find(v) == node_set.end())
				{
					node_queue.push(make_pair(graph->nodes[v], v));
					node_set.insert(v);
				}
			}
		}
	}
	auto stop = high_resolution_clock::now(); 

	// Print shortest distances stored in dist[] 
	printf("Vertex Distance from Source\n"); 
	for (int i = 0; i < graph->nodes.size(); ++i) 
		printf("%d \t\t %d\n", i, dist[i]); 

	auto duration = duration_cast<milliseconds>(stop - start);
	cout << "duration :" << duration.count() << endl;
	return; 
} 

shared_ptr<Graph> create_graph()
{
	shared_ptr<Graph> graph = make_shared<Graph>();

	fstream fin;
	fin.open("../matlab/gr.csv", ios::in);

	vector<int> row;
	string line, word;
	getline(fin,line);

	while (!fin.eof())
	{
		row.clear();
		getline(fin, line);
		stringstream s(line);

		while (getline(s, word, ','))  
		{
			row.push_back(stoi(word));
		}
		graph->addEdge(row[0]-1, row[1]-1, row[2]);
	}

	return graph;
}

// Driver program to test above functions 
int main()
{ 
	shared_ptr<Graph> graph;
	graph = create_graph();

	BellmanFord(graph, 312);

	return 0; 
} 
