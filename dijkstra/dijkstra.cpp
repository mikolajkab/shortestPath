// Program to find Dijkstra's shortest path using 
// priority_queue in STL 
#include<bits/stdc++.h> 
#include <chrono>
#include <fstream>

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

// Prints shortest paths from src to all other vertices 
void shortestPath(shared_ptr<Graph> graph, int src) 
{
	vector<int> dist(graph->nodes.size(), INT_MAX); 

	// Create a priority queue to store vertices that 
    // are being preprocessed. This is weird syntax in C++. 
    // Refer below link for details of this syntax 
    // https://www.geeksforgeeks.org/implement-min-heap-using-stl/
	priority_queue< iPair, vector <iPair> , greater<iPair> > pq; 

	// Insert source itself in priority queue and initialize 
	// its distance as 0.
	pq.push(make_pair(0, src)); 
	dist[src] = 0; 

	list<iPair>::iterator i;
	
	/* Looping till priority queue becomes empty (or all 
	distances are not finalized) */
	auto start = high_resolution_clock::now();
	while (!pq.empty()) 
	{ 
		// The first vertex in pair is the minimum distance 
		// vertex, extract it from priority queue. 
		// vertex label is stored in second of pair (it 
		// has to be done this way to keep the vertices 
		// sorted distance (distance must be first item 
		// in pair) 
		int u = pq.top().second; 
		pq.pop(); 

		// 'i' is used to get all adjacent vertices of a vertex 
		for (i = graph->nodes[u].begin(); i != graph->nodes[u].end(); ++i) 
		{ 
			// Get vertex label and weight of current adjacent 
			// of u. 
			int v = (*i).first; 
			int weight = (*i).second; 

			// If there is shorted path to v through u. 
			if (dist[v] > dist[u] + weight) 
			{ 
				// Updating distance of v 
				dist[v] = dist[u] + weight; 
				pq.push(make_pair(dist[v], v)); 
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

	auto duration = duration_cast<milliseconds>(stop - start);
	cout << "duration :" << duration.count() << endl;
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

// Driver program to test methods of graph class 
int main() 
{ 
	shared_ptr<Graph> graph;
	graph = create_graph();

	shortestPath(graph, 0);

	return 0; 
} 
