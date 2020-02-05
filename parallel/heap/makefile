all: heap

heap: heap.o
	 g++ -fopenmp -o heap heap.o

heap.o: heap.cpp
	 g++ -fopenmp -c heap.cpp
     
clean:
	 rm heap.o heap
