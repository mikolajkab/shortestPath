all: critical

critical: critical.o
	 g++ -fopenmp -o critical critical.o

critical.o: critical.cpp
	 g++ -fopenmp -c critical.cpp
     
clean:
	 rm critical.o critical
