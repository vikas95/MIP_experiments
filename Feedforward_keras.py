

# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset



def feedforward_keras(data, Y, data_test, Y_test, candidates, test_ans):
    # X = numpy.loadtxt(aligment_csv_file)
    X = data
    #dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
    ## Binary class classification
    # print("Input length is: ", data[0].size, data[0:20], Y[0:20], Y_test[0:20])
    model = Sequential()

    model.add(Dense(data[0].size, input_dim=data[0].size, activation="sigmoid"))
    model.add(Dense(60, activation='relu'))

    model.add(Dropout(0.20))

    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    # Fit the model
    model.fit(X, Y, epochs=10, batch_size=8) #  validation_split=0.5     validation_data = (data_test, Y_test)
    # evaluate the model
    scores = model.evaluate(data_test, Y_test)
    predictions1 = model.predict(data_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    # print("the predictions are: ",predictions1[0:20])
    Question_accuracy=0
    ques_score_pred = []
    gold_score = []

    ques_num=0
    upper_bound = len(candidates[ques_num])

    tot_cands = 0
    for cand123 in candidates:
        tot_cands+=len(cand123)

    for ind, val in enumerate(predictions1):

        if ind == upper_bound:
           # print(ques_score_pred)
           pred_label = ques_score_pred.index(max(ques_score_pred))
           # print(pred_label)
           gold_label = gold_score.index(max(gold_score))


           if pred_label == test_ans[ques_num]:
              # print(gold_label)
              Question_accuracy += 1

           ques_score_pred = []
           gold_score = []

           ques_score_pred += [val[0]]
           gold_score.append(Y_test[ind])

           ques_num+=1
           upper_bound+=len(candidates[ques_num])
           # print(len(candidates[ques_num]))
        elif ind == len(predictions1)-1:
            pred_label = ques_score_pred.index(max(ques_score_pred))
            # print(pred_label)
            gold_label = gold_score.index(max(gold_score))

            if pred_label == test_ans[ques_num]:
                Question_accuracy += 1

            ques_score_pred = []
            gold_score = []
        else:
            # ques_score_pred.append(val)
            ques_score_pred += [val[0]]
            gold_score.append(Y_test[ind])

    print("total number of questions are: ", ques_num, len(test_ans))
    # print ("The final question accuracy is: ", (Question_accuracy*4)/float(len(Y_test)))
    return (Question_accuracy)/float(len(candidates))


    

