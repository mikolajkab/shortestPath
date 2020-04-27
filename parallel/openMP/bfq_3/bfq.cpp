// A C++ program for Bellman-Ford's queue-based algorithm. 
#include <bits/stdc++.h>
#include <chrono>
#include <fstream>
#include <omp.h>

using namespace std;
using namespace std::chrono;

#define INF 2000000000

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
	int dist[graph->nodes.size()];
	for (int i=0; i<graph->nodes.size(); i++)
        dist[i] = INF;

	bool in_queue[graph->nodes.size()];
	for (int i=0; i<graph->nodes.size(); i++)
        in_queue[i] = false;

	int came_from[graph->nodes.size()];
	for (int i=0; i<graph->nodes.size(); i++)
        came_from[i] = INF;

	dist[src] = 0;
	in_queue[src] = true;
	came_from[src] = src;

	int u;
	int v;
	int weight;
	int i;
	int tid;
	bool label_updated;
	int n_threads = 8;
	bool idle[n_threads] = {false, true, true, true, true, true, true, true};

	queue<int> node_queue;
	queue<int> queues[n_threads];
	queues[0].push(src);

	omp_lock_t lock[n_threads];

	for (int i=0; i<n_threads; i++)
        omp_init_lock(&(lock[i]));
		
	int n_nodes = 10000;
	omp_lock_t lock_dist[n_nodes];
	omp_lock_t lock_came_from[n_nodes];

	for (int i=0; i<n_nodes; i++)
        omp_init_lock(&(lock_dist[i]));

	for (int i=0; i<n_nodes; i++)
        omp_init_lock(&(lock_came_from[i]));	

	// main loop
	auto start = high_resolution_clock::now(); 

	#pragma omp parallel private(tid, weight, label_updated, node_queue) shared(idle, queues) num_threads(n_threads)
	{
		tid = omp_get_thread_num();

		while(!(idle[0] && idle[1] && idle[2] && idle[3] && idle[4] && idle[5] && idle[6] && idle[7]))
		{
			if (queues[tid].empty())
			{
				idle[tid] = true;
			}
			else
			{
				idle[tid] = false;
				int u = queues[tid].front();
				queues[tid].pop();

				in_queue[u] = false;

				vector<int> not_in_queue;

				for (int i = 0; i < graph->nodes[u].size(); ++i)
				{
					int v = graph->nodes[u][i].first;
					weight = graph->nodes[u][i].second;

					if (dist[v] > dist[u] + weight)
					{
						int temp = dist[u] + weight;
						omp_set_lock(&(lock_dist[v]));
						{
							dist[v] = temp;
							came_from[v] = u;
						}
						omp_unset_lock(&(lock_dist[v]));
						
						if(!in_queue[v])
						{
							not_in_queue.push_back(v);
						}
					}
				}

				if(!not_in_queue.empty())
				{	
					int min_size = queues[0].size();
					int min_index;
					
					for (int j = 0; j < n_threads; ++j)
					{
						int temp_size = queues[j].size();
						if(temp_size == 0)
						{
							min_index = j;
							break;
						}

						if (temp_size < min_size)
						{
							min_size = temp_size;
							min_index = j;
						}
					}
					// printf("min_index: %d\n", min_index);
			
					for(std::vector<int>::iterator it = not_in_queue.begin(); it != not_in_queue.end(); ++it) 
					{
						if(!in_queue[*it])
						{
							omp_set_lock(&(lock[min_index]));
							{	
								queues[min_index].push(*it);
							}
							omp_unset_lock(&(lock[min_index]));

							in_queue[*it] = true;
						}
					}
				}
			}
		}
	}

	auto stop = high_resolution_clock::now(); 
	printf("test\n");

	for (int i=0; i<n_threads; i++)
        omp_destroy_lock(&(lock[i]));

	for (int i=0; i<n_nodes; i++)
        omp_destroy_lock(&(lock_dist[i]));

	for (int i=0; i<n_nodes; i++)
        omp_destroy_lock(&(lock_came_from[i]));

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
		cout << "total: " << total << "\n";

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
