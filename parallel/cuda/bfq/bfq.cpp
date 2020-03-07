// A C++ program for Bellman-Ford's queue-based single source 
// shortest path algorithm. 
#include <bits/stdc++.h>
#include <chrono>
#include <fstream>
#include <omp.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

const string fin_str = "../../../matlab/gr_10000_100.csv";

typedef pair<int, int> iPair; 

// This class represents a directed graph
class Graph 
{ 
public:
	Graph();

	void addEdge(int u, int v, int w);
	
	vector<vector<iPair> > nodes; 
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

// The main function that finds shortest distances
void BellmanFord(shared_ptr<Graph> graph, int src, int goal) 
{ 
	vector<int> dist(graph->nodes.size(), INT_MAX);
	vector<bool>in_queue(graph->nodes.size(), false);
	vector<int> came_from(graph->nodes.size(), INT_MAX);

	dist[src] = 0;
	in_queue[src] = true;
	came_from[src] = src;

	queue<int> node_queue;
	node_queue.push(src);

	// main loop
	auto start = high_resolution_clock::now(); 
	while(!node_queue.empty())
	{
		int u = node_queue.front();
		node_queue.pop();
		in_queue[u] = false;

		#pragma omp parallel shared(u, dist, came_from, node_queue, in_queue, graph)
		{
			#pragma omp for schedule(static) nowait
			for (int i = 0; i < graph->nodes[u].size(); ++i)
			{
				int v = graph->nodes[u][i].first;
				int weight = graph->nodes[u][i].second;

				if (dist[v] > dist[u] + weight)
				{
					#pragma omp critical
					{
						if (dist[v] > dist[u] + weight)
						{
							dist[v] = dist[u] + weight;
							came_from[v] = u;

							if (!in_queue[v])
							{
								in_queue[v] = true;
								node_queue.push(v);
							}
						}
					}
				}
			}
		}
	}
	auto stop = high_resolution_clock::now(); 

	// Print shortest distances stored in dist[] 
	ofstream myfile ("bfq.txt");
  	if (myfile.is_open())
  	{
		for (int i = 0; i < graph->nodes.size(); ++i) 
			myfile << i << "\t\t" << dist[i] <<"\n"; 
    	myfile.close();
  	}
  	else cout << "Unable to open file";

	ofstream myfile_path ("bfq_path.txt");
	if (myfile_path.is_open())
	{
		vector<int> path;
		int current = goal;
		while(current != src)
		{
			path.push_back(current);
			current = came_from[current];
		}
		path.push_back(src);
		reverse(path.begin(), path.end());

		for (vector<int>::iterator i = path.begin(); i < path.end(); ++i)
		{
			myfile_path << *i << "\t\t";
		}
    	myfile_path.close();
	} 
  	else cout << "Unable to open file";

	auto duration = duration_cast<milliseconds>(stop - start);
	cout << "duration :" << duration.count() << endl;
} 

shared_ptr<Graph> create_graph()
{
	shared_ptr<Graph> graph = make_shared<Graph>();

	fstream fin;
	fin.open(fin_str, ios::in);

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
	fin.close();

	return graph;
}

// Driver program to test above functions 
int main()
{ 
	shared_ptr<Graph> graph;
	graph = create_graph();

	BellmanFord(graph, 0, 10);

	return 0; 
} 
