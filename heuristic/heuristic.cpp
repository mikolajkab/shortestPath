// Program to generate heuristic for a given graph
#include <fstream>
#include <stdlib.h>     /* srand, rand */
#include<bits/stdc++.h> 

using namespace std;

#define INF 2000000000

const string fin_str = "../matlab/gr_optimal_control_3rd_order.csv";
const string fout_str = "../matlab/h_optimal_control_3rd_order.csv";

typedef pair<float, int> fiPair; 

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

void generate_heuristic(shared_ptr<Graph> graph, int src)
{
	/* initialize random seed: */
	srand(time(NULL));

	// open file
	fstream fout;
	fout.open(fout_str, ios::out);

	vector<float> dist(graph->nodes.size(), INF);
	vector<float> heuristic(graph->nodes.size(), INF);
	priority_queue< fiPair, vector <fiPair> , greater<fiPair> > pq; 

	dist[src] = 0;
	heuristic[src] = 0;
	pq.push(make_pair(0, src));

	while(!pq.empty())
	{
		int u = pq.top().second; 
		pq.pop();

		for (int i = 0; i < graph->nodes[u].size(); ++i)
		{
			int v = graph->nodes[u][i].second;
			float weight = graph->nodes[u][i].first;

			if(dist[v] > dist[u] + weight)
			{
				dist[v] = dist[u] + weight;
				pq.push(make_pair(dist[v], v));
				float r = ((float) rand() / (RAND_MAX));
				heuristic[v] = r * weight + dist[u];
				// cout << "weight: " << weight << ", r: " << r << endl;
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

	generate_heuristic(graph, 2324);

	return 0;
} 