// A C++ program for Bellman-Ford's queue-based algorithm. 
#include <bits/stdc++.h>
#include <chrono>
#include <fstream>
#include <omp.h>

using namespace std;
using namespace std::chrono;

#define INF 2000000000

const string fin_str = "../../../matlab/gr_optimal_control_3rd_order.csv";

typedef pair<float, int> fiPair; 

// This class represents a directed graph
class Graph 
{ 
public: 
	Graph(); 

	void addEdge(int u, int v, float w); 

	vector<vector<fiPair> > nodes; 
}; 

Graph::Graph() 
{ 
} 

void Graph::addEdge(int u, int v, float w)
{ 
	if (u >= nodes.size())
	{
		nodes.resize(u+1);
	}
	if (v >= nodes.size())
	{
		nodes.resize(v+1);
	}

	nodes[u].push_back(make_pair(w, v)); 
	nodes[v].push_back(make_pair(w, u));
}

// The main function that finds shortest distances
void BellmanFord(shared_ptr<Graph> graph, int src, int goal) 
{ 
	vector<float> dist(graph->nodes.size(), INF);
	vector<bool>in_queue(graph->nodes.size(), false);
	vector<int> came_from(graph->nodes.size(), INF);

	dist[src] = 0;
	in_queue[src] = true;
	came_from[src] = src;

	deque<int> node_queue;
	node_queue.push_front(src);

	bool idle[8]; 

	// main loop
	auto start = high_resolution_clock::now(); 

	#pragma omp parallel shared(idle) num_threads(8)
	{
		int tid = omp_get_thread_num();
		int u;

		while(!(idle[0] && idle[1] && idle[2] && idle[3] && idle[4] && idle[5] && idle[6] && idle[7]))
		{
			if (node_queue.empty())
			{
				idle[tid] = true;
			}
			else
			{
				#pragma omp critical
				{
					if(!node_queue.empty())
					{
						u = node_queue.front();
						node_queue.pop_front();
						idle[tid] = false;
					}
				}

				if(!idle[tid])
				{
					in_queue[u] = false;

					for (int i = 0; i < graph->nodes[u].size(); ++i)
					{
						int v = graph->nodes[u][i].second;
						float weight = graph->nodes[u][i].first;

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
					}
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

	ofstream myfile_path ("slf_path.txt");
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

		float total = 0;

		for (vector<int>::iterator i = path.begin(); i < path.end()-1;)
		{
			int u = *i;
			int v = *(++i);
			float weight = 0.0;
			for(int j = 0; j < graph->nodes[u].size()-1; ++j)
			{
				if (graph->nodes[u][j].second == v)
				{
					weight = graph->nodes[u][j].first;
					break;
				}
			}
			total += weight;
			cout << "u: " << u << ", v: " << v <<  ", weight: " << weight << "\n";
		}
		cout << "total: " << total <<"\n";
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

	vector<float> row;
	string line, word;
	getline(fin,line);

	while (!fin.eof())
	{
		row.clear();
		getline(fin, line);
		stringstream s(line);

		while (getline(s, word, ','))  
		{
			row.push_back(stof(word));
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

	BellmanFord(graph, 0, 2324);

	return 0; 
} 
