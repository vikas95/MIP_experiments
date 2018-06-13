import heapq
import math
import ast
import numpy as np
from numpy import array
from collections import Counter
from nltk.tokenize import RegexpTokenizer  ### for nltk word tokenization
tokenizer = RegexpTokenizer(r'\w+')
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()
import os
import csv


from Preprocess_NN_BM25_alignment import Preprpcess_NN_data
# from Preprocess_SIGIR_BM25 import Preprpcess_NN_data

from Feedforward_keras import feedforward_keras

import nltk
from nltk.corpus import stopwords
stop_words=stopwords.words('english')
stop_words=[lmtzr.lemmatize(w1) for w1 in stop_words]
stop_words=list(set(stop_words))
# stop_words=[]
print(stop_words)
print(len(stop_words))
"""
Vocab_file=open("Vocab.txt","r")
for line1 in Vocab_file:
    All_words=ast.literal_eval(line1)
"""
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


def Word2Vec_score(Question, Q_term_list, PMI_vals, IDF_Mat, Corpus, IDF, Justification_threshold):

    Justification_set=[]
    Document_score=[[0] for i in range(len(Question))]
    Alignment_score = [[0] for i in range(len(Question))]
    BM25_scores = [] # [[0] for i in range(len(Question))]
    PMI_scores = [[0] for i in range(len(Question))]

    for Jind, Justifications in enumerate(Corpus):

        threshold_vals=1
        if Jind%1000==0:
           print (Jind)
           # print(threshold_vals)
        Justification_set = []
        BM25_set = [0]
        Justifications = Justifications.strip()
        cols = Justifications.split("\t")  ## cols[0] has the question number, cols[1]  has the candidate option number for that specific question.
        Feature_col = cols[6].split(";;")
        # print (len(Feature_col))
        if len(Feature_col) >= Justification_threshold:
            for ind1 in range(Justification_threshold):  #### we take only top 10 justifications.
                ##["AggregatedJustification"]["text"]
                dict1 = ast.literal_eval(Feature_col[ind1])
                Justification_set.append((dict1["AggregatedJustification"]["text"]).lower())
                #if ind1 == 0:
                   #pass
                #else:
                BM25_set.append(dict1["AggregatedJustification"]["score"])

        BM25_scores.append(BM25_set)
        max_PMI = [[0] for i in range(len(Q_terms_list[Jind]))]

        for just_ind, just1 in enumerate(Justification_set):
            Doc_set = tokenizer.tokenize(just1)
            # Doc_set=list(set(Doc_set))
            Doc_set = [lmtzr.lemmatize(w1) for w1 in Doc_set]
            Doc_set = [w for w in Doc_set if not w in stop_words]

            ## Calculating PMI scores here, we have justification terms from above and we have query terms from Q_term_list[Jind]

            for tind1, term in enumerate(Q_terms_list[Jind]):
                term_PMI = []
                for tj1 in Doc_set:
                    try:
                       term_PMI.append(float(PMI_vals[term + " " + tj1]))
                    except KeyError:
                       pass
                if len(term_PMI)>0:
                   max_PMI[tind1].append(max(term_PMI))


            Doc_Matrix = np.empty((0, emb_size), float)  ####################### DIMENSION of EMBEDDING
            Doc_len=0
            for key in Doc_set:
                if key in embeddings_index.keys():
                   Doc_Matrix=np.append(Doc_Matrix, np.array([embeddings_index[key]]), axis=0)
                   Doc_len+=1
            if Doc_Matrix.size==0:
               pass
            else:

                Doc_IDF_Mat = np.empty((0, 1), float)
                Doc_IDF_Mat_min = np.empty((0, 1), float)

                Q_term_Mat = np.empty((0, 1), float)

                Doc_Matrix=Doc_Matrix.transpose()
                #print(Doc_Matrix.shape)
                ques1=Question[Jind]
                #threshold_vals = math.ceil(0.75 * float(ques1.shape[0])) ## math.ceil
                threshold_vals = ques1.shape[0]

                Score=np.matmul(ques1,Doc_Matrix)
                max_indices = np.argmax(Score, axis=1)
                min_indices = np.argmin(Score, axis=1)
                max_list=[]
                for mind1 in max_indices:
                    if Doc_set[mind1] in IDF.keys():
                        Doc_IDF_Mat = np.append(Doc_IDF_Mat, np.array([[IDF[Doc_set[mind1]]]]), axis=0)
                        max_list.append(Doc_set[mind1])
                    else:
                        Doc_IDF_Mat = np.append(Doc_IDF_Mat, np.array([[5.379046132954042]]), axis=0)

                counter2=0
                for mind1 in min_indices:
                    if Doc_set[mind1] in IDF.keys():
                        Doc_IDF_Mat_min = np.append(Doc_IDF_Mat_min, np.array([[IDF[Doc_set[mind1]]]]), axis=0)
                    else:

                        counter2+=1
                        Doc_IDF_Mat_min= np.append(Doc_IDF_Mat_min, np.array([[5.379046132954042]]), axis=0)

                Score = np.sort(Score, axis=1)
                max_score1 = Score[:, -1:]
                # max_score=np.multiply(np.transpose(IDF_Mat[Jind]),max_score)
                max_score1 = np.multiply(IDF_Mat[Jind], max_score1)

                # max_score=(sum(max_score1))#.item(0) ## this is the original without any threshold on the values.
                max_score = []
                max_alignment_score=0
                for qind1, qword1 in enumerate(max_score1):
                    # max_val=0
                    # qword1=qword1[::-1]
                    max_val = qword1[-1]
                    for i1, s1 in enumerate(qword1):
                        max_score += [(s1 / float(i1 + 1))]
                    max_alignment_score = sum(max_score)
                #max_score_d= (sum(max_score_d))

                #print (max_score)
                min_score = Score[:, 0:1]
                min_score1 = np.multiply(IDF_Mat[Jind], min_score)
                #min_score_d = np.multiply(np.transpose(Doc_IDF_Mat_min), min_score)  ### Becky suggestion which is not working

                min_alignment_score = 0
                for qind1, qword1 in enumerate(min_score1):
                    # qword1 = qword1[::-1]

                    for i1, s1 in enumerate(qword1):
                        max_score += [(s1 / float(i1 + 1))]     ## i1 +
                        min_alignment_score += (s1 / float(i1 + 1))

                # total_alignment_score = max_alignment_score + 0.4*(min_alignment_score)  ## + max_score_d + min_score_d
                # total_score=total_score/float(ques1.shape[0])
                # print("check here for neg val: ",max_score)
                total_alignment_score = [max_alignment_score,min_alignment_score]
                Document_score[Jind].append(max_score)
                Alignment_score[Jind].append(total_alignment_score)

        PMI_scores[Jind]=max_PMI

    return Document_score, Alignment_score, BM25_scores, PMI_scores



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



def calculate_feature_matrix(Question_file, IDF):
    All_questions = []
    IDF_Mat=[]


    counter=0
    Correct_ans = []
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
        Q_terms_list.append(q_term1)
        Q_terms_list.append(q_term2)
        Q_terms_list.append(q_term3)
        Q_terms_list.append(q_term4)

    return All_questions, Q_terms_list, IDF_Mat, Correct_ans



################################################################# train_files

file2=open("IDF_doc.txt","r")
for line2 in file2:
    IDF=ast.literal_eval(line2)

PMI_values = open("PMI_8th_grade_ARISTO.txt","r")
for line3 in PMI_values:
    PMI_vals=ast.literal_eval(line3)


file1=open("structured_kerasInput_train_bestIR_08j5.tsv","r")
Question_file = open('training_set.tsv', 'r')
All_questions, Q_terms_list, IDF_Mat, Correct_ans = calculate_feature_matrix(Question_file, IDF)

# print ("size of Q_terms_lis is: ", len(Q_terms_list), len(Q_terms_list[0]))

J_Threshold=3
Score_matrix, Alignment_scores_train, BM25_scores_train, PMI_train = Word2Vec_score(All_questions, Q_terms_list, PMI_vals, IDF_Mat,  file1, IDF, J_Threshold)

print ("len of PMI train ", len(PMI_train), PMI_train[0][0], len(PMI_train[1]), len(PMI_train[2]))

print ("the correct ans is: ", Correct_ans)
query_score = []
accuracy = 0
for PMIind, PMI_ques_values in enumerate(PMI_train):
    tot_score=0

    if PMIind%100 == 0:
       print (PMIind)

    for arr1 in PMI_ques_values:
        tot_score+=sum(arr1)
    query_score.append(tot_score)

    if len(query_score) == 4:
       pred_cand = query_score.index(max(query_score))
       query_score = []
       if pred_cand == int(Correct_ans[int((PMIind+1)/float(4)) - 1]):
          accuracy+=1
       # act_answer = Correct_ans.index(max(Correct_ans))

print ("the final accuracy is: ", accuracy/float(len(Correct_ans)))


"""

print ("len of Alignment_scores train ", len(Score_matrix), len(Score_matrix[0]), Score_matrix[1][0], len(Score_matrix[2]))

data,labels=Preprpcess_NN_data(Score_matrix, Alignment_scores_train, BM25_scores_train, Correct_ans, J_Threshold).prepare_data()


## converting labels for multi-class Keras version
multiclass_label = []
max_class = max(labels)
for label in labels:
    dum_lab = [0 for i in range(int(max_class)+1)]

    dum_lab[int(label)] = 1
    multiclass_label+=(dum_lab)

Y=array(multiclass_label)
data = array(data)

# print(data[0])
#######################################################
## TEST Files
#######################################################


file3=open("IDF_test_doc.txt","r")
for line2 in file3:
    IDF_test=ast.literal_eval(line2)


file1=open("structured_kerasInput_dev_bestIR_08j5.tsv","r")
Question_file_test = open('test_set.tsv', 'r')
All_questions, Q_terms_list, IDF_Mat, Correct_ans = calculate_feature_matrix(Question_file_test, IDF_test)
J_Threshold=5
Score_matrix, Alignment_scores_test, BM25_scores_test, PMI_test = Word2Vec_score(All_questions, Q_terms_list, PMI_vals, IDF_Mat,  file1, IDF_test, J_Threshold)


data_test,labels_test=Preprpcess_NN_data(Score_matrix, Alignment_scores_test, BM25_scores_test, Correct_ans, J_Threshold).prepare_data()

## converting labels for multi-class Keras version
multiclass_label_test = []
max_class = max(labels_test)
for label in labels_test:
    dum_lab = [0 for i in range(int(max_class)+1)]

    dum_lab[int(label)] = 1
    multiclass_label_test+=(dum_lab)

Y_test = array(multiclass_label_test)
data_test = array(data_test)

# print(data_test[0])
# print(data_test[1])
#######################################################
### 5 FOLD CROSS VALIDATION

# perf1 = feedforward_keras(data, Y, data_test, Y_test)
# print ("Performance of fold 1", perf1)



folds = 5
interval = len(data) / float(folds)
interval = math.floor(interval)
total_performance = []
for i in range(folds):

    if i == 0:

       train_data = data[interval*(i + 1):]
       train_label = Y[interval*(i+1):]

       test_data = data[0:interval * (i+1)]
       test_label = Y[0:interval * (i+1)]

    elif i == folds-1:
       train_data = data[0:interval * (i)]
       train_label = Y[0:interval * (i)]

       test_data = data[interval * (i):]
       test_label = Y[interval * (i):]

    else:
       train_data = np.concatenate((data[0:interval * (i)], data[interval * (i+1):]),axis = 0)
       train_label = np.concatenate((Y[0:interval * (i)], Y[interval * (i+1):]), axis=0)

       test_data = data[interval * (i) : interval * (i+1)]
       test_label = Y[interval * (i) : interval * (i+1)]



    perf1 = feedforward_keras(train_data, train_label, test_data, test_label)
    print ("Performance of fold 1", perf1)
    total_performance.append(perf1)

print ("total performance is: ", sum(total_performance)/float(folds))


"""


