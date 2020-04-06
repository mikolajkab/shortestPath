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

const string fin_str = "../matlab/gr_1000_499.csv";

typedef pair<int, int> iPair;

__global__ void relax_initial(int * d_dist, int* h_came_from, bool* h_in_queue, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

	d_dist[i] = INT_MAX;

	if (i == 0) 
	{
		d_dist[i] = 0;

	}
	__syncthreads();
}

__global__ void bf(int n, int u, int const* d_weights, int* d_dist, bool* d_in_queue, int* came_from)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;

	int weight = d_weights[u * n + v];
	if (weight < INT_MAX)
	{
		if (d_dist[v] > d_dist[u] + weight)
		{
			d_dist[v] = d_dist[u] + weight;
			d_in_queue[v] = true;
			came_from[v] = u;
		}
	}
}

// The main function that finds shortest distances
void BellmanFord(int src, int goal, int n, int h_weights[]) 
{ 
	dim3 threadsPerBlock = 256;
	dim3 blocksPerGrid = ((n + threadsPerBlock.x - 1) / threadsPerBlock.x);
	
	// host 
	int *h_dist = (int *)calloc(sizeof(int), n);
	int *h_came_from = (int *)calloc(sizeof(int), n);
	bool *h_in_queue = (bool *)calloc(sizeof(bool), n);

	for (int i=0; i<n; i++)
	{
		h_dist[i] = INT_MAX;
		h_came_from[i] = INT_MAX;
		h_in_queue[i] = false;
	}

	h_dist[src] = 0;
	h_came_from[src] = src;
	h_in_queue[src] = true;

	// device
	int* d_weights;
	int* d_dist;
	int* d_came_from;
	bool* d_in_queue;

	cudaMalloc(&d_weights, n * n * sizeof(int));
	cudaMalloc(&d_dist, n * sizeof(int));
	cudaMalloc(&d_came_from, n * sizeof(int));
	cudaMalloc(&d_in_queue, n * sizeof(bool));

	// copy host to device
	cudaMemcpy(d_weights, h_weights, n * n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dist, h_dist, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_came_from, h_came_from, n * sizeof(int), cudaMemcpyHostToDevice);

	queue<int> node_queue;
	node_queue.push(src);

	// main loop
	auto start = high_resolution_clock::now();
	while(!node_queue.empty())
	{
		int u = node_queue.front();
		node_queue.pop();
		h_in_queue[u] = false;

		cudaMemcpy(d_in_queue, h_in_queue, n * sizeof(bool), cudaMemcpyHostToDevice);

        // invoke kernel
		bf <<<blocksPerGrid, threadsPerBlock>>>(n, u, d_weights, d_dist, d_in_queue, d_came_from);
	
		cudaMemcpy(h_in_queue, d_in_queue, n * sizeof(bool), cudaMemcpyDeviceToHost);

		for (int i=0; i<n; i++)
		{
			if (h_in_queue[i])
			{
				node_queue.push(i);
			}
		}
	}

	auto stop = high_resolution_clock::now(); 

	cudaMemcpy(h_dist, d_dist, sizeof(int) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_came_from, d_came_from, n * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_weights);
	cudaFree(d_dist);
	cudaFree(d_came_from);
	cudaFree(d_in_queue);

	// Print shortest distances stored in dist[] 
	ofstream myfile ("bfq.txt");
  	if (myfile.is_open())
  	{
		for (int i = 0; i < n; ++i) 
			myfile << i << "\t\t" << h_dist[i] <<"\n"; 
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
			current = h_came_from[current];
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

//translate 2-dimension coordinate to 1-dimension
int convert_dimension_2D_1D(int x, int y, int n) 
{
	return x * n + y;
}

void create_weights(int weights[], int n)
{
	for (int i = 0; i < n * n; i++) 
	{
		weights[i] = INT_MAX;
	}

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

		weights[convert_dimension_2D_1D(row[0]-1, row[1]-1, n)] = row[2];
	}
	fin.close();
}

// Driver program to test above functions 
int main()
{
	int N = 10;
	int* mat = (int *)malloc(N * N * sizeof(int));

	create_weights(mat, N);

	BellmanFord(0, 10, N, mat);

	return 0; 
} 
