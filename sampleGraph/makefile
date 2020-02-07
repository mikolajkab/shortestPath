all: sample

sample: sample.o
	 g++ -fopenmp -o sample sample.o

sample.o: sample.cpp
	 g++ -fopenmp -c sample.cpp
     
clean:
	 rm sample.o sample
