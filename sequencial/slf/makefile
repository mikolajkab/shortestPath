all: slf

slf: slf.o
	 g++ -fopenmp -o slf slf.o

slf.o: slf.cpp
	 g++ -fopenmp -c slf.cpp
     
clean:
	 rm slf.o slf
