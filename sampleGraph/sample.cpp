// file to generate sample graphs
#include <fstream>

using namespace std; 

const string fout_str = "../matlab/gr_sample.csv";

int main() 
{ 

    // open file
    fstream fout;
    fout.open(fout_str, ios::out);

	fout << "node_1,node_2,weight\n";

	for(int i = 10000; i >= 0; --i )
	{
		fout << 1 << "," << 2 << "," << i <<"\n";
	}
	
	fout.close();

	return 0; 
} 