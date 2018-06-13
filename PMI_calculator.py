PMI_numerator = open("Proximity_count_ARISTO_Processed_10.txt","r")
PMI_denominator = open("DF_Individual_terms_ARISTO_Processed.txt","r")
total_docs = 119446442

DF={}
for line in PMI_denominator:
    words = line.strip().split()
    DF.update({words[0]:int(words[1])})


joint_count = {}

out_Vocab = []
for line in PMI_numerator:
    words = line.strip().split()
    count = words[-1]
    sec_word = words[1].split("~")[0]
    first_term = words[0][1:]
    second_term = sec_word[0:-1]
    try:
       PMI_val = (total_docs*int(count))/(float(DF[first_term]*DF[second_term]))
       joint_count.update({first_term+" "+second_term:PMI_val})
    except KeyError:
       out_Vocab.append(first_term)


PMI_file=open("PMI_8th_grade_ARISTO.txt","w")
PMI_file.write(str(joint_count))