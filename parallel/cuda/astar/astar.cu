// A C++ program for Astar shortest path algorithm. 
#include <bits/stdc++.h>
#include <chrono>
#include <fstream>
#include <omp.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

#define INF 2000000000

const string fin_gr_str = "../matlab/gr_optimal_control_3rd_order.csv";
const string fin_h_str = "../matlab/h_optimal_control_3rd_order.csv";

typedef pair<float, int> fiPair; 

__global__ void bf(int n, int u, float const* d_weights, float* d_dist, bool* d_has_change, int* came_from)
{
	int v = blockIdx.x * blockDim.x + threadIdx.x;
	if(v < n)
	{
		d_has_change[v] = false;

		float weight = d_weights[u * n + v];
		if (weight < INF)
		{
			if (d_dist[v] > d_dist[u] + weight)
			{
				d_dist[v] = d_dist[u] + weight;
				d_has_change[v] = true;
				came_from[v] = u;
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
void Astar(int src, int goal, int n, float h_weights[], float heuristic[]) 
{ 
	dim3 threadsPerBlock = 256;
	dim3 blocksPerGrid = ((n + threadsPerBlock.x - 1) / threadsPerBlock.x);
	
	// host 
	float *h_dist = (float *)calloc(sizeof(float), n);
	int *h_came_from = (int *)calloc(sizeof(int), n);
	bool *h_has_change = (bool *)calloc(sizeof(bool), n);
	
	for (int i=0; i<n; i++)
	{
		h_dist[i] = INF;
		h_came_from[i] = INF;
	}

	h_dist[src] = 0;
	h_came_from[src] = src;

	// device
	float* d_weights;
	float* d_dist;
	int* d_came_from;
	bool* d_has_change;

	cudaMalloc(&d_weights, n * n * sizeof(float));
	cudaMalloc(&d_dist, n * sizeof(float));
	cudaMalloc(&d_came_from, n * sizeof(int));
	cudaMalloc(&d_has_change, n * sizeof(bool));

	// copy host to device
	cudaMemcpy(d_weights, h_weights, n * n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dist, h_dist, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_came_from, h_came_from, n * sizeof(int), cudaMemcpyHostToDevice);

	priority_queue< fiPair, vector <fiPair> , greater<fiPair> > pq; 
	pq.push(make_pair(heuristic[src], src));

	int counter = 0;
	// main loop
	auto start = high_resolution_clock::now();
	while(!pq.empty())
	{
		int u = pq.top().second; 
		pq.pop();

		if(u == goal)
		{
			break;
		}

		counter++;

        // invoke kernel
		bf <<<blocksPerGrid, threadsPerBlock>>>(n, u, d_weights, d_dist, d_has_change, d_came_from);
	
		cudaMemcpy(h_has_change, d_has_change, n * sizeof(bool), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_dist, d_dist, sizeof(float) * n, cudaMemcpyDeviceToHost);

		for (int i = 0; i < n; i++)
		{
			if (h_has_change[i])
			{
				pq.push(make_pair(h_dist[i] + heuristic[i], i)); 
			}
		}
	}

	auto stop = high_resolution_clock::now(); 

	cout << "counter: " << counter << "\n";

	cudaMemcpy(h_came_from, d_came_from, n * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_weights);
	cudaFree(d_dist);
	cudaFree(d_came_from);
	cudaFree(d_has_change);

	// Print shortest distances stored in dist[] 
	ofstream myfile ("astar.txt");
  	if (myfile.is_open())
  	{
		for (int i = 0; i < n; ++i) 
			myfile << i << "\t\t" << h_dist[i] <<"\n"; 
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
			current = h_came_from[current];
		}
		path.push_back(src);
		reverse(path.begin(), path.end());

		for (vector<int>::iterator i = path.begin(); i < path.end(); ++i)
		{
			myfile_path << *i << "\t\t";
		}
		myfile_path.close();
		
		float total = 0;
		for (vector<int>::iterator i = path.begin(); i < path.end()-1;)
		{
			int u = *i;
			int v = *(++i);
			float weight = h_weights[convert_dimension_2D_1D(u, v, n)];
			total += weight;
			cout << "u: " << u << ", v: " << v <<  ", weight: " << weight << "\n";
		}
		cout << "total: " << total <<"\n";
	} 
  	else cout << "Unable to open file";

	auto duration = duration_cast<milliseconds>(stop - start);
	cout << "duration :" << duration.count() << endl;
}

void create_weights(float weights[], int n)
{
	for (int i = 0; i < n * n; i++) 
	{
		weights[i] = INF;
	}

	fstream fin;
	fin.open(fin_gr_str, ios::in);

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

		weights[convert_dimension_2D_1D(row[0]-1, row[1]-1, n)] = row[2];
		weights[convert_dimension_2D_1D(row[1]-1, row[0]-1, n)] = row[2];
	}
	fin.close();
}

void create_heuristic(float heuristic[], int n)
{
	for (int i = 0; i < n; i++) 
	{
		heuristic[i] = INF;
	}

	fstream fin_h;
	fin_h.open(fin_h_str, ios::in);

	vector<float> row;
	vector<int> row_int;

	string line, word;
	
	// don`t process the first line with column names
	getline(fin_h,line);

	// add heuristic to graph
	while (!fin_h.eof())
	{
		row.clear();
		row_int.clear();
		getline(fin_h, line);
		stringstream s(line);

		while (getline(s, word, ','))
		{
			row.push_back(stof(word));
			row_int.push_back(stoi(word));
		}

		heuristic[row_int[0]] = row[2];
	}

	fin_h.close();
}

// Driver program to test above functions 
int main()
{
	int N = 16456;
	float* mat = (float *)malloc(N * N * sizeof(float));
	float* heuristic = (float *)malloc(N * sizeof(float));

	create_weights(mat, N);
	create_heuristic(heuristic, N);

	// for (int i=0; i< N*N; i++)
	// {
	// 	cout << mat[i] << " ";
	// }

	Astar(0, 2324, N, mat, heuristic);

	return 0; 
} 
