all: dijkstra

dijkstra: dijkstra.cu
	 /usr/local/cuda-10.2/bin/nvcc -o dijkstra dijkstra.cu
     
clean:
	 rm dijkstra.o dijkstra
