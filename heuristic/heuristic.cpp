// Program to generate heuristic for a given graph
#include <fstream>
#include <stdlib.h>     /* srand, rand */
#include<bits/stdc++.h> 

using namespace std; 

const string fin_str = "../matlab/gr_10000_1000.csv";
const string fout_str = "../matlab/h_10000_1000.csv";

// iPair ==> Integer Pair 
typedef pair<int, int> iPair; 

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

void generate_heuristic(shared_ptr<Graph> graph, int src)
{
	/* initialize random seed: */
	srand(time(NULL));

	// open file
	fstream fout;
	fout.open(fout_str, ios::out);

	vector<int> dist(graph->nodes.size(), INT_MAX);
	vector<int> heuristic(graph->nodes.size(), INT_MAX);
	priority_queue< iPair, vector <iPair> , greater<iPair> > pq; 

	dist[src] = 0;
	heuristic[src] = 0;
	pq.push(make_pair(0, src));

	while(!pq.empty())
	{
		int u = pq.top().second; 
		pq.pop();

		for (int i = 0; i < graph->nodes[u].size(); ++i)
		{
			int v = graph->nodes[u][i].first;
			int weight = graph->nodes[u][i].second;

			if(dist[v] > dist[u] + weight)
			{
				dist[v] = dist[u] + weight;
				pq.push(make_pair(dist[v], v));
				heuristic[v] = rand() % weight + dist[u] + 1;
				// heuristic[v] = dist[v];
			}
		}
	}

	fout << "node,distance,heuristic\n";
	for(int i = 0; i < dist.size(); ++i )
	{
		fout << i << "," << dist[i] << "," << heuristic[i] <<"\n";
	}
	
	fout.close();
}


int main() 
{ 
	shared_ptr<Graph> graph;
	graph = create_graph();

	generate_heuristic(graph, 10);

	return 0; 
} 