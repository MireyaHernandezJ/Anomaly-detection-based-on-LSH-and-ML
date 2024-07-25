import csv
import pickle
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from collections import Counter
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

def count_ceros(arr):
    count=0
    for num in arr:
        if num==0:
               count+=1
    return count 

#DATASET:

dataset_path="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/Datasets_5s/dataset_binary.pkl"
with open(dataset_path,'rb') as f:
           dataset=pickle.load(f)


labels_path="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/Datasets_5s/labels_binary.pkl"
with open(labels_path,'rb') as f:
           labels=pickle.load(f)


train,test,ltrain,ltest=train_test_split(dataset,labels,test_size=0.2,random_state=42)

normal_train_samples=count_ceros(ltrain)
attack_train_samples=len(ltrain)-count_ceros(ltrain)

normal_test_samples=count_ceros(ltest)
attack_test_samples=len(ltest)-count_ceros(ltest)


data="FPS_WifiMQTT-TrainTestNEW-Binary"
hash="nilsimsa" #tlsh or nilsimsa
windowtime="5s"#["5s","10s","15s","20s"]

 #SECTION - S I M U L A T I O N 2: DECISION TREE

# model1 = "DT"
# classifier1 = DecisionTreeClassifier(random_state=0,max_depth=3)
# hyperparameters1 = "max_depth=3"


# model2 = "DT"
# classifier2 = DecisionTreeClassifier(random_state=0,max_depth=4)
# hyperparameters2 = "max_depth=5"


# model3 = "DT"
# classifier3 = DecisionTreeClassifier(random_state=0,max_depth=5)
# hyperparameters3 = "max_depth=9"

# model4 = "DT"
# classifier4 = DecisionTreeClassifier(random_state=0,max_depth=6)
# hyperparameters4 = "max_depth=40"

# classifiers = [classifier1, classifier2, classifier3, classifier4]
# models = [model1, model2, model3, model4]
# hyperparameterss= [hyperparameters1, hyperparameters2, hyperparameters3, hyperparameters4]

#SECTION - S I M U L A T I O N 3: RANDOM FOREST
# model1 = "RF"
# classifier1 = RandomForestClassifier(random_state=0,max_depth=29,n_estimators=250)
# hyperparameters1 = "max_depth,Nestimators=29,250"


# model2 = "RF"
# classifier2 = RandomForestClassifier(random_state=0,max_depth=10,n_estimators=175)
# hyperparameters2 = "max_depth,Nestimators=10,175"


# model3 = "RF"
# classifier3 = RandomForestClassifier(random_state=0,max_depth=32,n_estimators=25)
# hyperparameters3 ="max_depth,Nestimators=32,25"

# model4 = "RF"
# classifier4 = RandomForestClassifier(random_state=0,max_depth=37,n_estimators=225)
# hyperparameters4 = "max_depth,Nestimators=37,225"


# model5 = "RF"
# classifier5 = RandomForestClassifier(random_state=0,max_depth=12,n_estimators=250)
# hyperparameters5 = "max_depth,Nestimators=12,250"

# classifiers = [classifier1, classifier2, classifier3, classifier4,classifier5]
# models = [model1, model2, model3, model4,model5]
# hyperparameterss= [hyperparameters1, hyperparameters2, hyperparameters3, hyperparameters4,hyperparameters5]



#SECTION - S I M U L A T I O N 4: Logistic Regression
# model1 = "LR"
# classifier1 = LogisticRegression(random_state=0,max_iter=160)
# hyperparameters1 = "max_iter=160"

# classifiers = [classifier1]
# models = [model1]
# hyperparameterss= [hyperparameters1]


#SECTION - S I M U L A T I O N 5:
model1 = "Adaboost"
classifier1 = AdaBoostClassifier(random_state=0,n_estimators=125)
hyperparameters1 = "Nestimators=125"

model2 = "Adaboost"
classifier2 = AdaBoostClassifier(random_state=0,n_estimators=150)
hyperparameters2 = "Nestimators=150"

model3 = "Adaboost"
classifier3 = AdaBoostClassifier(random_state=0,n_estimators=225)
hyperparameters3 = "Nestimators=225"

classifiers = [classifier1, classifier2, classifier3]
models = [model1, model2, model3]
hyperparameterss= [hyperparameters1, hyperparameters2, hyperparameters3]


################## S A V I N G

for idx, classifier in enumerate(classifiers):
    samples_train=len(train)
    samples_test=len(test)

    #Modeling
    start=time.time()
    clf=classifier.fit(train,ltrain)
    stop=time.time()
    training_time=stop-start

    pred=clf.predict(test)

    acc=accuracy_score(ltest,pred)*100
    prec_attack=precision_score(ltest,pred)*100
    rec_attack=recall_score(ltest,pred,pos_label=1)*100
    f1_attack=f1_score(ltest,pred,pos_label=1)*100
    AUCs=roc_auc_score(ltest,pred)*100
    tn,fp,fn,tp=confusion_matrix(ltest,pred).ravel()
    FPRs=fp/(tn+fp)*100


    new_row=[
        data,
        attack_train_samples,
        normal_train_samples,
        attack_test_samples,
        normal_test_samples,
        samples_train,
        samples_test,
        models[idx],
        hyperparameterss[idx],
        hash,
        windowtime,
        tn,
        fp,
        fn,
        tp,
        acc,
        prec_attack,
        rec_attack,
        f1_attack,
        AUCs,
        FPRs,
        training_time,
    ]

    # Open the CSV file in append mode
    with open(
            "/home/mireya/Documentos/Paper_3/Results_CICIoMT2024/results_WIFI-MQTT_%s.csv"%(model1),
            mode="a",
            newline="",
    ) as csvfile:
        # Create a CSV writer
        writer = csv.writer(csvfile)
        # Write the new row to the CSV file
        writer.writerow(new_row)

