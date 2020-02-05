all: hello

hello: hello.o
	 g++ -fopenmp -o hello hello.o

hello.o: hello.cpp
	 g++ -fopenmp -c hello.cpp
     
clean:
	 rm hello.o hello
