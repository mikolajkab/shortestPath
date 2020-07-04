// Program to find shortest path using Dijkstra algorithm
#include<bits/stdc++.h> 
#include <chrono>
#include <fstream>

using namespace std; 
using namespace std::chrono;

#define INF 2000000000

const string fin_str = "../../matlab/gr_10000_4000.csv";

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

// Prints shortest paths from src to all other vertices 
void shortestPath(shared_ptr<Graph> graph, int src, int goal) 
{
	vector<int> dist(graph->nodes.size(), INF); 
	vector<int> came_from(graph->nodes.size(), INF);
	priority_queue< iPair, vector <iPair> , greater<iPair> > pq;

	dist[src] = 0;
	came_from[src] = src;
	pq.push(make_pair(0, src)); 

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
		
		for (int i = 0; i < graph->nodes[u].size(); ++i)
		{ 
			int v = graph->nodes[u][i].first; 
			int weight = graph->nodes[u][i].second;

			if (dist[v] > dist[u] + weight) 
			{ 
				dist[v] = dist[u] + weight; 
				pq.push(make_pair(dist[v], v)); 
				came_from[v] = u;
			} 
		} 
	} 
	auto stop = high_resolution_clock::now(); 

	// Print shortest distances stored in dist[]
	ofstream myfile ("dijkstra.txt");
  	if (myfile.is_open())
  	{
		for (int i = 0; i < graph->nodes.size(); ++i) 
			myfile << i << "\t\t" << dist[i] <<"\n"; 
    	myfile.close();
  	}
  	else cout << "Unable to open file";

	ofstream myfile_path ("dijkstra_path.txt");
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

		int total = 0;

		for (vector<int>::iterator i = path.begin(); i < path.end()-1;)
		{
			int u = *i;
			int v = *(++i);
			int weight = 0;
			for(int j = 0; j < graph->nodes[u].size()-1; ++j)
			{
				if (graph->nodes[u][j].first == v)
				{
					weight = graph->nodes[u][j].second;
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

// Driver program to test methods of graph class 
int main() 
{ 
	shared_ptr<Graph> graph;
	graph = create_graph();

	shortestPath(graph, 0, 10);

	return 0; 
} 
