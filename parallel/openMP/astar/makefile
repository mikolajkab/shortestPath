all: astar

astar: astar.o
	 g++ -fopenmp -o astar astar.o

astar.o: astar.cpp
	 g++ -fopenmp -c astar.cpp
     
clean:
	 rm astar.o astar
