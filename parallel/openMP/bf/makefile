all: bf

bf: bf.o
	 g++ -fopenmp -o bf bf.o

bf.o: bf.cpp
	 g++ -fopenmp -c bf.cpp
     
clean:
	 rm bf.o bf
