// Program to find shortest path using astar algorithm 
#include<bits/stdc++.h> 
#include <chrono>
#include <fstream>
#include <stdlib.h>     /* srand, rand */
#include <omp.h>

using namespace std; 
using namespace std::chrono;

#define INF 2000000000

const string fin_gr_str = "../../../matlab/gr_optimal_control_3rd_order.csv";
const string fin_h_str = "../../../matlab/h_optimal_control_3rd_order.csv";

typedef pair<float, int> fiPair; 

// This class represents a directed graph
class Graph 
{ 
public: 
	Graph(); 

	void addEdge(int u, int v, float w); 
	void addHeuristic(int u, float h); 

	vector<vector<fiPair> > nodes; 
	vector<float> heuristic;
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

void Graph::addHeuristic(int u, float h)
{
	if (u >= heuristic.size())
	{
		heuristic.resize(u+1);
	}
	heuristic[u] = h;
}

// Prints shortest paths from src to goal 
void shortestPath(shared_ptr<Graph> graph, int src, int goal) 
{
	vector<float> dist(graph->nodes.size(), INF); 
	vector<int> came_from(graph->nodes.size(), INF);
	vector<float> heuristic = graph->heuristic;
	priority_queue< fiPair, vector <fiPair> , greater<fiPair> > pq;

	dist[src] = 0; 
	came_from[src] = src;
	pq.push(make_pair(0 + heuristic[src], src)); 
	
	/* Looping till priority queue becomes empty */
	auto start = high_resolution_clock::now();
	while (!pq.empty()) 
	{
		int u = pq.top().second; 
		pq.pop(); 

		if(u == goal)
		{
			break;
		}

		#pragma omp parallel shared(u, dist, came_from, pq, heuristic)
		{
			#pragma omp for schedule(static) nowait
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
							pq.push(make_pair(dist[v] + heuristic[v], v)); 
						} 
					}
				}
			}
		}
	}
	
	auto stop = high_resolution_clock::now(); 

	// Print shortest distances stored in dist[]
	ofstream myfile ("astar.txt");
  	if (myfile.is_open())
  	{
		for (int i = 0; i < graph->nodes.size(); ++i) 
			myfile << i << "\t\t" << dist[i] <<"\n"; 
    	myfile.close();
  	}
  	else cout << "Unable to open file";

	ofstream myfile_path ("astar_path.txt");
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

	fstream fin_gr, fin_h;
	fin_gr.open(fin_gr_str, ios::in);
	fin_h.open(fin_h_str, ios::in);

	vector<float> row;
	string line, word;

	// dont process the first line with column names
	getline(fin_gr,line);
	
	// generate graph
	while (!fin_gr.eof())
	{
		row.clear();
		getline(fin_gr, line);
		stringstream s(line);

		while (getline(s, word, ','))  
		{
			row.push_back(stof(word));
		}

		graph->addEdge(row[0]-1, row[1]-1, row[2]);
	}

	// don`t process the first line with column names
	getline(fin_h,line);
	
	// add heuristic to graph
	while (!fin_h.eof())
	{
		row.clear();
		getline(fin_h, line);
		stringstream s(line);

		while (getline(s, word, ','))  
		{
			row.push_back(stof(word));
		}

		graph->addHeuristic(row[0], row[2]);
	}
	fin_h.close();
	fin_gr.close();

	return graph;
}

// Driver program to test methods of graph class 
int main() 
{ 
	shared_ptr<Graph> graph;
	graph = create_graph();

	shortestPath(graph, 0, 2324);

	return 0; 
}