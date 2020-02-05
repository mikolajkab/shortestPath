all: sections

sections: sections.o
	 g++ -fopenmp -o sections sections.o

sections.o: sections.cpp
	 g++ -fopenmp -c sections.cpp
     
clean:
	 rm sections.o sections
