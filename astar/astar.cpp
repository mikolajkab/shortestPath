// Program to find shortest path using astar algorithm 
#include<bits/stdc++.h> 
#include <chrono>
#include <fstream>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

using namespace std; 
using namespace std::chrono;

// iPair ==> Integer Pair 
typedef pair<int, int> iPair; 

// This class represents a directed graph using 
// adjacency list representation 
class Graph 
{ 
public: 
	Graph(); 

	void addEdge(int u, int v, int w); 
	void addHeuristic(int u, int h); 

	vector<list<iPair> > nodes;
	vector<int> heuristic;
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

void Graph::addHeuristic(int u, int h)
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
	vector<int> dist(graph->nodes.size(), INT_MAX); 
	vector<int> came_from(graph->nodes.size(), INT_MAX);
	vector<int> heuristic = graph->heuristic;

	// Create a priority queue to store vertices that 
    // are being preprocessed.
	priority_queue< iPair, vector <iPair> , greater<iPair> > pq; 

	// Insert source itself in priority queue and initialize 
	// its distance to its heuristic.
	pq.push(make_pair(0 + heuristic[src], src)); 
	dist[src] = 0; 
	came_from[src] = src;

	list<iPair>::iterator i;
	
	/* Looping till priority queue becomes empty (or all 
	distances are not finalized) */
	auto start = high_resolution_clock::now();
	while (!pq.empty()) 
	{ 
		// The vertex in the first pair is the minimum distance 
		// vertex, extract it from priority queue. 
		// vertex label is stored in second of pair
		int u = pq.top().second; 
		pq.pop(); 

		if(u == goal)
		{
			break;
		}

		// 'i' is used to get all adjacent vertices of a vertex 
		for (i = graph->nodes[u].begin(); i != graph->nodes[u].end(); ++i) 
		{ 
			// Get vertex label and weight of current adjacent of u. 
			int v = (*i).first; 
			int weight = (*i).second; 

			// If there is a shorter path to v through u. 
			if (dist[v] > dist[u] + weight) 
			{ 
				// Update distance of v 
				dist[v] = dist[u] + weight; 
				pq.push(make_pair(dist[v] + heuristic[v], v));
				came_from[v] = u;
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
	} 
  	else cout << "Unable to open file";

	auto duration = duration_cast<microseconds>(stop - start);
	cout << "duration :" << duration.count() << endl;
} 

shared_ptr<Graph> create_graph()
{
	shared_ptr<Graph> graph = make_shared<Graph>();

	/* initialize random seed: */
	srand (time(NULL));

	fstream fin_gr, fin_h;
	fin_gr.open("../matlab/gr_100000_5.csv", ios::in);
	fin_h.open("../matlab/h_100000_5.csv", ios::in);

	vector<int> row;
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
			row.push_back(stoi(word));
		}

		graph->addEdge(row[0]-1, row[1]-1, row[2]);
	}

	int u = 0;
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
			row.push_back(stoi(word));
		}

		graph->addHeuristic(row[0], row[2]);

		u++;
	}

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
