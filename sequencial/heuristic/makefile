all: heuristic

heuristic: heuristic.o
	 g++ -fopenmp -o heuristic heuristic.o

heuristic.o: heuristic.cpp
	 g++ -fopenmp -c heuristic.cpp
     
clean:
	 rm heuristic.o heuristic
