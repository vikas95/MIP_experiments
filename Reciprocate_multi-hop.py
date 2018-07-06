import heapq
import math
import ast
import numpy as np
from collections import Counter
from nltk.tokenize import RegexpTokenizer  ### for nltk word tokenization
tokenizer = RegexpTokenizer(r'\w+')
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()
import os


import nltk
from nltk.corpus import stopwords
stop_words=stopwords.words('english')

# stop_words = ["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
# "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these", "they", "this", "to", "was", "will", "with"] ## Lucene stopwords...


stop_words=[lmtzr.lemmatize(w1) for w1 in stop_words]
stop_words=list(set(stop_words))
# stop_words=[]
print(stop_words)
print(len(stop_words))

# becky_emb=open("ss_qz_04.dim50vecs.txt","r", encoding='utf-8')
embeddings_index = {}
# glove_emb = open('glove.6B.100d.txt','r', encoding='utf-8')
# f = open(os.path.join("/Users/vikasy/Glove_vectors/","glove.840B.300d.txt"),'r', encoding='utf-8')
# f = open(os.path.join("/Users/vikasy/Glove_vectors/","glove.840B.300d.txt"),'r', encoding='utf-8')
# f = open(os.path.join("/Users/vikasy/Glove_vectors/","wiki.english.vec"),'r', encoding='utf-8')
f = open('all_emb_Glove.txt','r', encoding='utf-8')

#f = open('ss_qz_04.dim50vecs.txt')
for line in f:
    values = line.split()
    word = values[0]
    # word = lmtzr.lemmatize(word)
    try:
       coefs = np.asarray(values[1:], dtype='float32')
       b = np.linalg.norm(coefs, ord=2)
       coefs=coefs/b
       emb_size=coefs.shape[0]
    except ValueError:
       print (values[0])
       continue
    embeddings_index[word] = coefs
print("Word2vc matrix len is : ",len(embeddings_index))
print("Embedding size is: ", emb_size)


file2=open("IDF_doc.txt","r")
for line2 in file2:
    IDF=ast.literal_eval(line2)



def Word2Vec_score(Question, Q_term_list, IDF_Mat, Corpus, IDF, Justification_threshold):

    Doc_Score=[0]


    max_score=0
    min_score=0
    #Ques_score=[]
    Justification_set=[]
    Document_score=[[0] for i in range(len(Question))]
    Justification_ind = [[0] for i in range(len(Question))]
    #SCORES=[]


    for Jind, Justifications in enumerate(Corpus):

        threshold_vals=1
        if Jind%1000==0:
           print (Jind)
           # print(threshold_vals)

        Justification_set = []
        Justifications = Justifications.strip()
        cols = Justifications.split("\t")  ## cols[0] has the question number, cols[1]  has the candidate option number for that specific question.
        Feature_col = cols
        # print (len(Feature_col))
        if len(Feature_col) >= Justification_threshold:
            for ind1 in range(Justification_threshold):  #### we take only top 10 justifications.
                ##["AggregatedJustification"]["text"]

                Justification_set.append(Feature_col[ind1].lower())
        """
        Justification_set = []
        Justifications = Justifications.strip()
        cols = Justifications.split("\t")  ## cols[0] has the question number, cols[1]  has the candidate option number for that specific question.
        Feature_col = cols[6].split(";;")
        # print (len(Feature_col))
        if len(Feature_col) >= Justification_threshold:
            for ind1 in range(Justification_threshold):  #### we take only top 10 justifications.
                ##["AggregatedJustification"]["text"]
                dict1 = ast.literal_eval(Feature_col[ind1])
                Justification_set.append((dict1["AggregatedJustification"]["text"]).lower())
        """

        for just_ind, just1 in enumerate(Justification_set):
            Doc_set = tokenizer.tokenize(just1)
            # Doc_set=list(set(Doc_set))
            Doc_set = [lmtzr.lemmatize(w1) for w1 in Doc_set]
            Doc_set = [w for w in Doc_set if not w in stop_words]

            Doc_Matrix = np.empty((0, emb_size), float)  ####################### DIMENSION of EMBEDDING
            Doc_len=0
            for key in Doc_set:
                if key in embeddings_index.keys():
                   Doc_Matrix=np.append(Doc_Matrix, np.array([embeddings_index[key]]), axis=0)
                   Doc_len+=1
            if Doc_Matrix.size==0:
               pass
            else:

                Q_term_Mat = np.empty((0, 1), float)

                Doc_Matrix=Doc_Matrix.transpose()
                ques1=Question[Jind]

                Score=np.matmul(ques1,Doc_Matrix)

                Score = np.sort(Score, axis=1)
                max_score1 = Score[:, -1:]
                max_score1 = np.multiply(IDF_Mat[Jind], max_score1)

                # max_score=(sum(max_score1))#.item(0) ## this is the original without any threshold on the values.
                max_score = 0

                for qind1, qword1 in enumerate(max_score1):
                    # max_val=0
                    # qword1=qword1[::-1]
                    max_val = qword1[-1]
                    for i1, s1 in enumerate(qword1):
                        max_score += (s1 / float(i1 + 1))
                #max_score_d= (sum(max_score_d))

                #print (max_score)
                min_score = Score[:, 0:1]
                min_score1 = np.multiply(IDF_Mat[Jind], min_score)
                #min_score_d = np.multiply(np.transpose(Doc_IDF_Mat_min), min_score)  ### Becky suggestion which is not working

                min_score = 0
                for qind1, qword1 in enumerate(min_score1):
                    # qword1 = qword1[::-1]

                    for i1, s1 in enumerate(qword1):
                        min_score += (s1 / float(i1 + 1))  ## i1 +


                total_score=max_score + 0.3*(min_score)  ## + max_score_d + min_score_d
                total_score=total_score/float(ques1.shape[0])
                Document_score[Jind].append(total_score)
                Justification_ind[Jind].append(just_ind)


    return Document_score, Justification_ind



def Ques_Emb(ques1, IDF):
    query_term=[]
    Ques_Matrix = np.empty((0, emb_size), float)
    IDF_Mat = np.empty((0, 1), float)  ##### IDF is of size = 1 coz its a value
    for q_term in ques1:
        if q_term in embeddings_index.keys():
           Ques_Matrix = np.append(Ques_Matrix, np.array([embeddings_index[q_term]]), axis=0)
           IDF_Mat = np.append(IDF_Mat, np.array([[IDF[q_term]]]), axis=0)
           query_term.append(q_term)

    return Ques_Matrix, IDF_Mat, query_term


Question_file = open('Query_expansion_hop1.tsv', 'r')
 #[]
Correct_ans = []#[]
All_questions = []
IDF_Mat=[]


counter=0
file1=open("8th_grade_CAND_question_explanations_3_BM25.txt","r")
# file1 = open("8th_grade_CAND_question_explanations_3_BM25.txt","r")

Final_scores=[]
All_terms=[]
All_Ques_terms=[]
Q_terms_list=[]

for line1 in Question_file:
    counter+=1
    #print(counter)
    Question = ""
    Option_A = ""  # []  ####### These will contain justification text also and later on, becky features will be added.
    Option_B = ""  # []
    Option_C = ""  # []
    Option_D = ""
    Cand_score = []
    line1 = line1.strip()
    cols = line1.split("\t")
    Correct_ans.append(cols[3])
    A_index = cols[10].index("(A)")
    B_index = cols[10].index("(B)")
    C_index = cols[10].index("(C)")
    D_index = cols[10].index("(D)")

    Question = (cols[10][:A_index - 1])
    Option_A = (cols[10][A_index + 4:B_index - 1])
    Option_B = (cols[10][B_index + 4:C_index - 1])
    Option_C = (cols[10][C_index + 4:D_index - 1])
    Option_D = (cols[10][D_index + 4:])

    Question = tokenizer.tokenize(Question.lower())
    Question=[lmtzr.lemmatize(w1) for w1 in Question]
    Question = [w for w in Question if not w in stop_words]

    Option_A = tokenizer.tokenize(Option_A.lower())
    Option_A = [lmtzr.lemmatize(w1) for w1 in Option_A]
    Option_A = [w for w in Option_A if not w in stop_words]

    Option_B = tokenizer.tokenize(Option_B.lower())
    Option_B = [lmtzr.lemmatize(w1) for w1 in Option_B]
    Option_B = [w for w in Option_B if not w in stop_words]

    Option_C = tokenizer.tokenize(Option_C.lower())
    Option_C = [lmtzr.lemmatize(w1) for w1 in Option_C]
    Option_C = [w for w in Option_C if not w in stop_words]

    Option_D = tokenizer.tokenize(Option_D.lower())
    Option_D = [lmtzr.lemmatize(w1) for w1 in Option_D]
    Option_D = [w for w in Option_D if not w in stop_words]

    All_Ques_terms=Question+Option_A+Option_B+Option_C+Option_D
    All_Ques_terms=list(set(All_Ques_terms))
    All_terms.append(All_Ques_terms)
    All_Ques_terms=[]


    Ques1 = Question + Option_A  ###### Question + Candidate answer 1
    Q1_matrix, IDF_Mat1, q_term1=Ques_Emb(Ques1, IDF)

    Ques2 = Question + Option_B
    Q2_matrix, IDF_Mat2, q_term2 = Ques_Emb(Ques2, IDF)

    Ques3 = Question + Option_C
    Q3_matrix, IDF_Mat3, q_term3 = Ques_Emb(Ques3, IDF)

    Ques4 = Question + Option_D
    Q4_matrix, IDF_Mat4, q_term4 = Ques_Emb(Ques4, IDF)




    All_questions += [Q1_matrix, Q2_matrix, Q3_matrix, Q4_matrix]
    IDF_Mat += [IDF_Mat1, IDF_Mat2, IDF_Mat3, IDF_Mat4]
    Q_terms_list += [q_term1, q_term2, q_term3, q_term4]


J_Threshold=5
Score_matrix, Justification_matrix = Word2Vec_score(All_questions, Q_terms_list, IDF_Mat,  file1, IDF, J_Threshold)

# print(Score_matrix)
out_file_name="dumm123"+".txt"
out_file=open(out_file_name,"w")
out_file.write(str(Score_matrix))


"""
Score_matrix, Justification_matrix = Word2Vec_score(All_questions, IDF_Mat,  file1, 3)
out_file=open("Becky_files_W2V_score_"+str(3)+"Final"+".txt","w")
out_file.write(str(Score_matrix))

"""
## Calculating accuracy here in the same file.
import ast
import numpy as np
from statistics import mean
out_file_name="dumm123"+".txt"
file1=open(out_file_name,"r")
scores=[]

for line in file1:
    scores=ast.literal_eval(line)

print("Score matrix is:  ")
print ("Are we reaching here at all ?? ")
# print (Score_matrix)

scores = Score_matrix

ind_score=[]
All_score=[]
Predicted_ans=[]
for ind1, s1 in enumerate(scores):
    # print ("we are entering this scores loop")
    ind_score.append(max(s1))
    All_score.append(sum(s1))
    if ind1%4==3:
       ind_score=np.asarray(ind_score)

       Predicted_ans.append(np.argmax(ind_score))
       ind_score=[]

# print("We are getting at line 346 ", All_score)
# print ("We are getting at line 347 ",Predicted_ans)

Question_file = open('training_set.tsv', 'r')
Correct_ans = []#[]
counter=0
for line1 in Question_file:
    counter += 1
    if counter>2500:
       break
    if counter<1:
       pass
    else:
        line1 = line1.strip()
        cols = line1.split("\t")
        Correct_ans.append(cols[3])


print(Correct_ans)

Accuracy=0
if len(Correct_ans)==len(Predicted_ans):
   for Pind, Pred1 in enumerate(Predicted_ans):

       if Pred1==int(Correct_ans[Pind]):
          Accuracy+=1

print("Accuracy for all ques is: ",str(Accuracy/float(len(Predicted_ans))))

