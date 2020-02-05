void update_recode() 
{ 

	// File pointer 
	fstream fin, fout; 

	// Open an existing record 
	fin.open("reportcard.csv", ios::in); 

	// Create a new file to store updated data 
	fout.open("reportcardnew.csv", ios::out); 

	int rollnum, roll1, marks, count = 0, i; 
	char sub; 
	int index, new_marks; 
	string line, word; 
	vector<string> row; 

	// Get the roll number from the user 
	cout << "Enter the roll number "
		<< "of the record to be updated: "; 
	cin >> rollnum; 

	// Get the data to be updated 
	cout << "Enter the subject "
		<< "to be updated(M/P/C/B): "; 
	cin >> sub; 

	// Determine the index of the subject 
	// where Maths has index 2, 
	// Physics has index 3, and so on 
	if (sub == 'm' || sub == 'M') 
		index = 2; 
	else if (sub == 'p' || sub == 'P') 
		index = 3; 
	else if (sub == 'c' || sub == 'C') 
		index = 4; 
	else if (sub == 'b' || sub == 'B') 
		index = 5; 
	else { 
		cout << "Wrong choice.Enter again\n"; 
		update_record(); 
	} 

	// Get the new marks 
	cout << "Enter new marks: "; 
	cin >> new_marks; 

	// Traverse the file 
	while (!fin.eof()) { 

		row.clear(); 

		getline(fin, line); 
		stringstream s(line); 

		while (getline(s, word, ', ')) { 
			row.push_back(word); 
		} 

		roll1 = stoi(row[0]); 
		int row_size = row.size(); 

		if (roll1 == rollnum) { 
			count = 1; 
			stringstream convert; 

			// sending a number as a stream into output string 
			convert << new_marks; 

			// the str() converts number into string 
			row[index] = convert.str(); 

			if (!fin.eof()) { 
				for (i = 0; i < row_size - 1; i++) { 

					// write the updated data 
					// into a new file 'reportcardnew.csv' 
					// using fout 
					fout << row[i] << ", "; 
				} 

				fout << row[row_size - 1] << "\n"; 
			} 
		} 
		else { 
			if (!fin.eof()) { 
				for (i = 0; i < row_size - 1; i++) { 

					// writing other existing records 
					// into the new file using fout. 
					fout << row[i] << ", "; 
				} 

				// the last column data ends with a '\n' 
				fout << row[row_size - 1] << "\n"; 
			} 
		} 
		if (fin.eof()) 
			break; 
	} 
	if (count == 0) 
		cout << "Record not found\n"; 

	fin.close(); 
	fout.close(); 

	// removing the existing file 
	remove("reportcard.csv"); 

	// renaming the updated file with the existing file name 
	rename("reportcardnew.csv", "reportcard.csv"); 
} 
