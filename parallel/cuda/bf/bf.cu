// A C++ program for Bellman-Ford's single source 
// shortest path algorithm. 
#include <bits/stdc++.h>
#include <chrono>
#include <fstream>
#include <omp.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

#define INF 2000000000

const string fin_str = "../matlab/gr_optimal_control_3rd_order.csv";

__global__ void bf(int n, int const* d_weights, int* d_dist, bool* d_has_change, int* came_from)
{
	int v = blockIdx.x * blockDim.x + threadIdx.x;

	if (v == 0)
		*d_has_change = false;
	__syncthreads();

	if(v >= n)
	{
		return;
	}

	for (int u = 0; u < n; ++u)
	{
		if (u != v)
		{
			int weight = d_weights[u * n + v];
			if (weight < INF)
			{
				if (d_dist[v] > d_dist[u] + weight)
				{
					d_dist[v] = d_dist[u] + weight;
					*d_has_change = true;
					came_from[v] = u;
				}
			}
		}
	}
	__syncthreads();
}

void bf_func(int n, int const* d_weights, int* d_dist, bool* d_has_change, int* came_from)
{
	*d_has_change = false;

	for (int v = 0; v < n; ++v)
	{
		for (int u = 0; u < n; ++u)
		{
			if (u != v)
			{
				int weight = d_weights[u * n + v];
				if (weight < INF)
				{
					if (d_dist[v] > d_dist[u] + weight)
					{
						d_dist[v] = d_dist[u] + weight;
						*d_has_change = true;
						came_from[v] = u;
						cout << "v: " << v << ", u: " << u << ", dist_v: " << d_dist[v] << ", dist_u: " << d_dist[u] << ", weight: " << weight << "\n";
					}
				}
			}
		}
	}
}

//translate 2-dimension coordinate to 1-dimension
int convert_dimension_2D_1D(int x, int y, int n) 
{
	return x * n + y;
}

// The main function that finds shortest distances
void BellmanFord(int src, int goal, int n, int h_weights[]) 
{ 
	dim3 threadsPerBlock = 256;
	dim3 blocksPerGrid = ((n + threadsPerBlock.x - 1) / threadsPerBlock.x);
	
	// host 
	int *h_dist = (int *)calloc(sizeof(int), n);
	int *h_came_from = (int *)calloc(sizeof(int), n);
	bool *h_has_change = (bool *)calloc(sizeof(bool), 1);
	
	for (int i=0; i<n; i++)
	{
		h_dist[i] = INF;
		h_came_from[i] = INF;
	}

	h_dist[src] = 0;
	h_came_from[src] = src;

	// device
	int* d_weights;
	int* d_dist;
	int* d_came_from;
	bool* d_has_change;

	cudaMalloc(&d_weights, n * n * sizeof(int));
	cudaMalloc(&d_dist, n * sizeof(int));
	cudaMalloc(&d_came_from, n * sizeof(int));
	cudaMalloc(&d_has_change, sizeof(bool));

	// copy host to device
	cudaMemcpy(d_weights, h_weights, n * n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dist, h_dist, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_came_from, h_came_from, n * sizeof(int), cudaMemcpyHostToDevice);

	// for(int i=0; i<n; i++)
	// {
	// 	cout << h_dist[i] << " ";
	// }
	// cout << "\n";

	int counter = 0;
	// main loop
	auto start = high_resolution_clock::now();
	while(true)
	{
		counter++;

        // invoke kernel
		bf <<<blocksPerGrid, threadsPerBlock>>>(n, d_weights, d_dist, d_has_change, d_came_from);
	
		// bf_func(n, h_weights, h_dist, h_has_change, h_came_from);

		cudaMemcpy(h_has_change, d_has_change, sizeof(bool), cudaMemcpyDeviceToHost);

		if(!(*h_has_change))
		{
			break;
		}
	}

	auto stop = high_resolution_clock::now(); 

	cout << "counter: " << counter << "\n";

	cudaMemcpy(h_dist, d_dist, n * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_came_from, d_came_from, n * sizeof(int), cudaMemcpyDeviceToHost);

	// for(int i=0; i<n; i++)
	// {
	// 	cout << h_dist[i] << " ";
	// }
	// cout << "\n";
	// for(int i=0; i<n; i++)
	// {
	// 	cout << h_came_from[i] << " ";
	// }
	// cout << "\n";

	cudaFree(d_weights);
	cudaFree(d_dist);
	cudaFree(d_came_from);
	cudaFree(d_has_change);

	// Print shortest distances stored in dist[] 
	ofstream myfile ("bf.txt");
  	if (myfile.is_open())
  	{
		for (int i = 0; i < n; ++i) 
			myfile << i << "\t\t" << h_dist[i] <<"\n"; 
    	myfile.close();
  	}
  	else cout << "Unable to open file";

	ofstream myfile_path ("bf_path.txt");
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
		
		int total = 0;
		for (vector<int>::iterator i = path.begin(); i < path.end()-1;)
		{
			int u = *i;
			int v = *(++i);
			int weight = h_weights[convert_dimension_2D_1D(u, v, n)];
			total += weight;
			cout << "u: " << u << ", v: " << v <<  ", weight: " << weight << "\n";
		}
		cout << "total: " << total <<"\n";
	} 
  	else cout << "Unable to open file";

	auto duration = duration_cast<milliseconds>(stop - start);
	cout << "duration :" << duration.count() << endl;
} 

void create_weights(int weights[], int n)
{
	for (int i = 0; i < n * n; i++) 
	{
		weights[i] = INF;
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
		weights[convert_dimension_2D_1D(row[1]-1, row[0]-1, n)] = row[2];
	}
	fin.close();
}

// Driver program to test above functions 
int main()
{
	int N = 16456;
	int* mat = (int *)malloc(N * N * sizeof(int));

	create_weights(mat, N);

	// for(int i=0; i<N*N; i++)
	// {
	// 	cout << i << ": "<< mat[i] << "\n";
	// }
	// cout << "\n";

	BellmanFord(0, 2324, N, mat);

	return 0; 
} 
