all: dijkstra

dijkstra: dijkstra.o
	 g++ -fopenmp -o dijkstra dijkstra.o

dijkstra.o: dijkstra.cpp
	 g++ -fopenmp -c dijkstra.cpp
     
clean:
	 rm dijkstra.o dijkstra
