// A C++ program for Bellman-Ford's queue-based single source 
// shortest path algorithm. 
#include <bits/stdc++.h> 
#include <chrono>
#include <fstream>

using namespace std;
using namespace std::chrono;

typedef pair<int, int> iPair; 

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
	// Step 1: Initialize distances from src to all other vertices 
	// as INFINITE 
	vector<int> dist(graph->nodes.size(), INT_MAX);
	vector<bool>in_queue(graph->nodes.size(), false);

	dist[src] = 0;

	deque<int> node_queue;
	node_queue.push_front(src);
	in_queue[src] = true;

	list<iPair>::iterator i;

	// Step 2: Relax all edges |V| - 1 times. A simple shortest 
	// path from src to any other vertex can have at-most |V| - 1 
	// edges
	auto start = high_resolution_clock::now(); 
	while(!node_queue.empty())
	{
		int u = node_queue.front();
		node_queue.pop_front();
		in_queue[u] = false;
		
		for (i = graph->nodes[u].begin(); i != graph->nodes[u].end(); ++i)
		{
			int v = (*i).first;
			int weight = (*i).second;

			if (dist[v] > dist[u] + weight) 
			{
				dist[v] = dist[u] + weight;

				if (!in_queue[v])
				{
					if(node_queue.empty() || dist[v] <= dist[node_queue.front()])
                    {
                        node_queue.push_front(v);
                    }
                    else
                    {
                        node_queue.push_back(v);
                    }
					in_queue[v] = true;
				}
			}
		}
	}
	auto stop = high_resolution_clock::now();

	// Print shortest distances stored in dist[] 
	ofstream myfile ("slf.txt");
  	if (myfile.is_open())
  	{
		for (int i = 0; i < graph->nodes.size(); ++i) 
			myfile << i << "\t\t" << dist[i] <<"\n"; 
    	myfile.close();
  	}
  	else cout << "Unable to open file";

	auto duration = duration_cast<milliseconds>(stop - start);
	cout << "duration :" << duration.count() << endl;

	return;
} 

shared_ptr<Graph> create_graph()
{
	shared_ptr<Graph> graph = make_shared<Graph>();

	// graph->addEdge(0, 1, 4);
	// graph->addEdge(0, 7, 8);
	// graph->addEdge(1, 2, 8);
	// graph->addEdge(1, 7, 11);
	// graph->addEdge(2, 3, 7);
	// graph->addEdge(2, 8, 2);
	// graph->addEdge(2, 5, 4);
	// graph->addEdge(3, 4, 9);
	// graph->addEdge(3, 5, 14);
	// graph->addEdge(4, 5, 10);
	// graph->addEdge(5, 6, 2);
	// graph->addEdge(6, 7, 1);
	// graph->addEdge(6, 8, 6);
	// graph->addEdge(7, 8, 7);

	fstream fin;
	fin.open("../matlab/gr_100000_5.csv", ios::in);

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

	BellmanFord(graph, 0);

	return 0;
} 
