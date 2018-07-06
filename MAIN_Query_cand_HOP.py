
"""
This is the main file for running Query to Query+Cand_Multi-hop

"""

embedding_file = open('all_emb_Glove.txt','r', encoding='utf-8')

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
       coefs = coefs / b
       emb_size=coefs.shape[0]
    except ValueError:
       print (values[0])
       continue
    embeddings_index[word] = coefs
print("Word2vc matrix len is : ",len(embeddings_index))
print("Embedding size is: ", emb_size)




from Preprocess_dataset import preprocess_dataset, preprocess_justifications
from hop_lexical_overlap import get_query_hop
from Alignment_function import Word2Vec_score




# embedding_file = open(os.path.join("/Users/vikasy/Glove_vectors/","glove.840B.300d.txt"),'r', encoding='utf-8')
file2=open("IDF_doc.txt","r")
for line2 in file2:
    IDF=ast.literal_eval(line2)



Question_file = open('training_set.tsv', 'r')
Query_cand_terms = preprocess_dataset(Question_file)

print(len(Query_cand_terms["Question_terms"]), len(Query_cand_terms["Cand_terms"]))

Just_Query_just = open("TRAIN_8th_grade_JUST_question_explanations_BM25.txt","r")

Query_justifications = preprocess_justifications(Just_Query_just)

# query_hop_justification_1 = Lexical_hop(Query_justifications, Query_cand_terms["Question"], Query_cand_terms["Cand_terms"])


query_hop1 = get_query_hop(Query_justifications, Query_cand_terms, 4 ).Lexical_hop()

print(len(query_hop1), len(query_hop1["Q_just_cand"][1]))

"""
Just verifying with actual query
"""
Query_hop_file1 = open("Query_expansion_hop1.txt","w")
for query_expans in query_hop1["Q_just_cand"]:
    Query_hop_file1.write(query_expans[0] + "\n")





question_file = open("TRAIN_8th_grade_Science_JUST_Questions.txt","r")
All_questions = []

for line in question_file:
    All_questions.append(line.strip())



# just_hop2_file = open("TRAIN_8th_grade_JUST_question_explanations_BM25.txt","r")
just_hop2_file = open("8th_grade_CAND_question_explanations_3_BM25.txt","r")
J_Threshold=2
# Score_matrix, Justification_matrix = Word2Vec_score(query_hop1["Q_just_cand"], just_hop2_file, IDF, J_Threshold, embeddings_index, emb_size)
Score_matrix, Justification_matrix = Word2Vec_score(All_questions, just_hop2_file, IDF, J_Threshold, embeddings_index, emb_size)

print("len of score matrix is ",len(Score_matrix), len(Score_matrix[1]))
print(Score_matrix)
# print(Score_matrix)
"""
len_verification = []
hop1_Vocab = []
for just in query_hop1["Q_just_cand"]:
    len_verification.append(len(just))
    for single_just in just:
        hop1_Vocab += single_just.split()

hop1_Vocab = list(set(hop1_Vocab))

print("set of len vectors : ", set(len_verification))


IDF_verification = []



noIDF = []

for term1 in hop1_Vocab:
    if term1 not in IDF.keys():
       noIDF.append(term1)

print("Out of Vocab IDF terms are: ", len(list(set(noIDF))), set(noIDF))
"""

scores = Score_matrix

ind_score=[]
All_score=[]
Predicted_ans=[]
for ind1, s1 in enumerate(scores):
    # print ("we are entering this scores loop")
    ind_score.append(sum(s1))
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

