// Program to generate heuristic for a given graph
#include <fstream>
#include <stdlib.h>     /* srand, rand */
#include<bits/stdc++.h> 

using namespace std; 

// iPair ==> Integer Pair 
typedef pair<int, int> iPair; 

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

shared_ptr<Graph> create_graph()
{
	shared_ptr<Graph> graph = make_shared<Graph>();

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

void generate_heuristic(shared_ptr<Graph> graph, int src)
{
	/* initialize random seed: */
	srand(time(NULL));

	// open file
	fstream fout;
	fout.open("../matlab/h_100000_5.csv", ios::out);

    vector<bool> visited(graph->nodes.size(), false);
	vector<int> dist(graph->nodes.size(), INT_MAX);
	vector<int> heuristic(graph->nodes.size(), INT_MAX);
	
	visited[src] = true;
	dist[src] = 0;
	heuristic[src] = 0;
	
	queue<int> q;
	q.push(src);

	while(!q.empty())
	{
		int u = q.front();
		q.pop();

		for(list<iPair>::iterator i = graph->nodes[u].begin(); i != graph->nodes[u].end(); ++i)
		{
			int v = (*i).first;
			int weight = (*i).second; 

			if(!visited[v])
			{
				visited[v] = true;
				q.push(v);
			}

			if(dist[v] > dist[u] + weight)
			{
				dist[v] = dist[u] + weight;
				heuristic[v] = rand() % weight + dist[u] + 1;
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

	generate_heuristic(graph, 0);

	return 0; 
} 