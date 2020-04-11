all: bfq

bfq: bfq.o
	 g++ -fopenmp -o bfq bfq.o

bfq.o: bfq.cpp
	 g++ -fopenmp -c bfq.cpp
     
clean:
	 rm bfq.o bfq
