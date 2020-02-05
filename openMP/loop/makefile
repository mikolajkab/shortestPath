all: loop

loop: loop.o
	 g++ -fopenmp -o loop loop.o

loop.o: loop.cpp
	 g++ -fopenmp -c loop.cpp
     
clean:
	 rm loop.o loop
