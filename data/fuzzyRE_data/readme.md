# FuzzyRE dataset
## data source
Please refer to the paper.
## data format
The data is prepared in Semeval10 format.  Each file has 50 instances of one relation whose id are contained in file name. Each instance consists of three lines: sentence, label and empty commnt. The sentence line has two columns, id (const value 1) and sentence. We use entity markers like <e1>, <e2> to mark the two entities in the sentence. There are two labels, `True` and `Other`. `True` means this sentence is a true positive case and `Other` means this sentence is a false positive case. 
 
