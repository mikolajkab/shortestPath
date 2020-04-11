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

const string fin_str = "../matlab/gr_1000_499.csv";

typedef pair<int, int> iPair;

__global__ void relax_initial(int * d_dist, int* h_came_from, bool* h_has_change, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

	d_dist[i] = INT_MAX;

	if (i == 0) 
	{
		d_dist[i] = 0;
	}
	__syncthreads();
}

__global__ void bf(int n, int const* d_weights, int* d_dist, bool* d_has_change, int* came_from)
{
	int v = blockIdx.x * blockDim.x + threadIdx.x;

	if(v < n)
	{
		*d_has_change = false;

		for (int u = 0; u < n; u++)
		{
			int weight = d_weights[u * n + v];
			if (weight < INT_MAX)
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
}

// The main function that finds shortest distances
void BellmanFord(int src, int goal, int n, int h_weights[]) 
{ 
	dim3 threadsPerBlock = 256;
	dim3 blocksPerGrid = ((n + threadsPerBlock.x - 1) / threadsPerBlock.x);
	
	// host 
	int *h_dist = (int *)calloc(sizeof(int), n);
	int *h_came_from = (int *)calloc(sizeof(int), n);
	bool h_has_change;
	
	for (int i=0; i<n; i++)
	{
		h_dist[i] = INT_MAX;
		h_came_from[i] = INT_MAX;
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

	int counter = 0;
	// main loop
	auto start = high_resolution_clock::now();
	while(true)
	{
		counter++;

        // invoke kernel
		bf <<<blocksPerGrid, threadsPerBlock>>>(n, d_weights, d_dist, d_has_change, d_came_from);
	
		cudaMemcpy(&h_has_change, d_has_change, sizeof(bool), cudaMemcpyDeviceToHost);

		if(!h_has_change)
		{
			break;
		}
	}

	cout << "counter: " << counter << "\n";

	auto stop = high_resolution_clock::now(); 

	cudaMemcpy(h_dist, d_dist, sizeof(int) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_came_from, d_came_from, n * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_weights);
	cudaFree(d_dist);
	cudaFree(d_came_from);
	cudaFree(d_has_change);

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
		weights[convert_dimension_2D_1D(row[1]-1, row[0]-1, n)] = row[2];
	}
	fin.close();
}

// Driver program to test above functions 
int main()
{
	int N = 1000;
	int* mat = (int *)malloc(N * N * sizeof(int));

	create_weights(mat, N);

	// for (int i=0; i< N*N; i++)
	// {
	// 	cout << mat[i] << " ";
	// }

	BellmanFord(0, 10, N, mat);

	return 0; 
} 
