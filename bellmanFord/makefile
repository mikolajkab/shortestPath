all: bellmanFord

bellmanFord: bellmanFord.o
	 g++ -fopenmp -o bellmanFord bellmanFord.o

bellmanFord.o: bellmanFord.cpp
	 g++ -fopenmp -c bellmanFord.cpp
     
clean:
	 rm bellmanFord.o bellmanFord
