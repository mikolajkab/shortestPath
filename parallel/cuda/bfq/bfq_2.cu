/*
* This is a CUDA version of bellman_ford algorithm
* Compile: nvcc -std=c++11 -arch=sm_52 -o cuda_bellman_ford cuda_bellman_ford.cu
* */

#include <string>
#include <cassert>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <ctime>


#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// #include "pnt.hpp"

using std::string;
using std::cout;
using std::endl;

#define INF 1000000

void pprint(int i, int n, bool stay = false)
{
	int p = (i + 1) * 100 / n;
	if (p != i * 100 / n)
		printf("%d%%\r", p);
	if (stay && p == 100)
		putchar('\n');
}

/*
* This is a CHECK function to check CUDA calls
*/
#define CHECK(call)                                                            \
		{                                                                              \
	const cudaError_t error = call;                                            \
	if (error != cudaSuccess)                                                  \
	{                                                                          \
		fprintf(stderr, "Error: %s%d, ", __FILE__, __LINE__);                 \
		fprintf(stderr, "code: %d, reason: %s\n", error,                       \
				cudaGetErrorString(error));                                    \
				exit(1);                                                               \
	}                                                                          \
		}


/**
* utils is a namespace for utility functions
* including I/O (read input file and print results) and matrix dimension convert(2D->1D) function
*/
namespace utils {
	int N; //number of vertices
	int *mat; // the adjacency matrix

	void abort_with_error_message(string msg) {
		std::cerr << msg << endl;
		abort();
	}

	//translate 2-dimension coordinate to 1-dimension
	int convert_dimension_2D_1D(int x, int y, int n) {
		return x * n + y;
	}

	int read_file(string filename) {
		std::ifstream inputf(filename, std::ifstream::in);
		if (!inputf.good()) {
			abort_with_error_message("ERROR OCCURRED WHILE READING INPUT FILE");
		}
		inputf >> N;
		//input matrix should be smaller than 20MB * 20MB (400MB, we don't have too much memory for multi-processors)
		assert(N < (1024 * 1024 * 20));
		mat = (int *)malloc(N * N * sizeof(int));
		printf("%d int malloced\n", N * N);
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				inputf >> mat[convert_dimension_2D_1D(i, j, N)];
			}
			pprint(i, N);
		}
		return 0;
	}

	int print_result(int *dist) {
		std::ofstream outputf("output.txt", std::ofstream::out);
			for (int i = 0; i < N; i++) {
				if (dist[i] > INF)
					dist[i] = INF;
				outputf << dist[i] << '\n';
			}
			outputf.flush();

		outputf.close();
		return 0;
	}
} //namespace utils

 // kernel functions

__global__ void relax_initial(int * d_dist, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

	d_dist[i] = INF;

	if (i == 0) 
	{
		d_dist[i] = 0;
	}
	__syncthreads();
}

__global__ void bf(int n, int const* d_mat, int * d_dist, bool * d_has_change)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;

	if (v == 0)
	{
		*d_has_change = false;
	}
	__syncthreads();

	bool my_has_change = false;

	if (v < n)
	{
		for (int u = 0; u < n; ++u) 
		{
			int weight = d_mat[u * n + v];
			if (weight < INF)
			{
				if (d_dist[u] + weight < d_dist[v]) 
				{
					d_dist[v] = d_dist[u] + weight;
					my_has_change = true;
				}
			}
		}
	}
	if (my_has_change)
		*d_has_change = true;
}

void bellman_ford(int n, int *h_dist, int *h_mat)
{
	dim3 threadsPerBlock(256);
	dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x);
	
	// host
	bool h_has_change = false;

	h_dist = (int *)calloc(sizeof(int), n);
	h_mat = (int *)calloc(sizeof(int), n * n);

	// device
	int *d_mat;
	int *d_dist;
	bool *d_has_change;

	cudaMalloc(&d_mat, n * n * sizeof(int));
	cudaMalloc(&d_dist, n * sizeof(int));
	cudaMalloc(&d_has_change, sizeof(bool));

	cudaMemcpy(d_mat, h_mat, n * n * sizeof(int), cudaMemcpyHostToDevice);

	relax_initial <<<blocksPerGrid, threadsPerBlock>>>(d_dist, n);

	while (true)
	{
		bf <<<blocksPerGrid, threadsPerBlock>>> (n, d_mat, d_dist, d_has_change);
		
		cudaMemcpy(&h_has_change, d_has_change, sizeof(bool), cudaMemcpyDeviceToHost);
		
		if (!h_has_change)
		{
			break;
		}
	}

	cudaMemcpy(h_dist, d_dist, sizeof(int) * n, cudaMemcpyDeviceToHost);

	cudaFree(d_mat);
	cudaFree(d_dist);
	cudaFree(d_has_change);
}

int main(int argc, char **argv) 
{
	int N = 100;

	bellman_ford(N, dist, mat);
	CHECK(cudaDeviceSynchronize());
	
	utils::print_result(dist);

	free(dist);
	free(mat);

	return 0;
}