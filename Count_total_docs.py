

import glob
import os

Total_lines=0
Short_lines=0

file1= open("Total_doc_ARISTO_processed.txt","w")
count = 0
for file_name in glob.glob(os.path.join(os.getcwd(), "ARISTO_processed/*.txt")):
    read_file = open(file_name,"r")
    count+=1
    print (count)
    for line in read_file:
        line = line.strip().split()
        if len(line)>1:
           Total_lines+=1
        else:
           Short_lines+=1


file1.write(str(Total_lines)+"\n")
file1.write(str(Short_lines))

