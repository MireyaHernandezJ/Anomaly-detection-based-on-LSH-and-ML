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

data="FPS_ICE-TrainTest-Binary"
hash="nilsimsa" 
windowtime="600"

dataset_path="/home/mireya/Documentos/Datasets-IoT/ICEdataset/ICEdataset_FPs_pps_binary/%s/dataset_%s.pkl"%(windowtime,windowtime)
with open(dataset_path,'rb') as f:
           dataset=pickle.load(f)

labels_path="/home/mireya/Documentos/Datasets-IoT/ICEdataset/ICEdataset_FPs_pps_binary/%s/labels_%s.pkl"%(windowtime,windowtime)
with open(labels_path,'rb') as f:
           labels=pickle.load(f)

train,test,ltrain,ltest=train_test_split(dataset,labels,test_size=0.2,random_state=42)

normal_test_samples=count_ceros(ltest)
attack_test_samples=len(ltest)-count_ceros(ltest)
normal_train_samples=count_ceros(ltrain)
attack_train_samples=len(ltrain)-count_ceros(ltrain)


#SECTION - S I M U L A T I O N 1

model1 = "LR"
classifier1 = LogisticRegression(random_state=0)
hyperparameters1= "default"

model2 = "DT"
classifier2 = DecisionTreeClassifier(random_state=0)
hyperparameters2 = "default"

model3 = "RF"
classifier3 = RandomForestClassifier(random_state=0)
hyperparameters3 = "default"


model4 = "Adaboost"
classifier4 = AdaBoostClassifier(random_state=0)
hyperparameters4 = "default"

classifiers = [classifier1, classifier2, classifier3, classifier4]
models = [model1, model2, model3, model4]
hyperparameterss= [hyperparameters1, hyperparameters2, hyperparameters3, hyperparameters4]


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
            "/home/mireya/Documentos/Paper_3/Results_ICEdataset/results_%s.csv"%(models[idx]),
            mode="a",
            newline="",
    ) as csvfile:
        # Create a CSV writer
        writer = csv.writer(csvfile)
        # Write the new row to the CSV file
        writer.writerow(new_row)


# #SECTION - S I M U L A T I O N 2: DECISION TREE

model1 = "DT"
classifier1 = DecisionTreeClassifier(random_state=0,max_depth=36)
hyperparameters1 = "max_depth=36"


model2 = "DT"
classifier2 = DecisionTreeClassifier(random_state=0,max_depth=6)
hyperparameters2 = "max_depth=6"


model3 = "DT"
classifier3 = DecisionTreeClassifier(random_state=0,max_depth=28)
hyperparameters3 = "max_depth=28"

model4 = "DT"
classifier4 = DecisionTreeClassifier(random_state=0,max_depth=40)
hyperparameters4 = "max_depth=40"

model5 = "DT"
classifier5 = DecisionTreeClassifier(random_state=0,max_depth=4)
hyperparameters5 = "max_depth=4"

classifiers = [classifier1, classifier2, classifier3, classifier4,classifier5]
models = [model1, model2, model3, model4,model5]
hyperparameterss= [hyperparameters1, hyperparameters2, hyperparameters3, hyperparameters4,hyperparameters5]




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
            "/home/mireya/Documentos/Paper_3/Results_ICEdataset/results_%s.csv"%(model1),
            mode="a",
            newline="",
    ) as csvfile:
        # Create a CSV writer
        writer = csv.writer(csvfile)
        # Write the new row to the CSV file
        writer.writerow(new_row)

#SECTION - S I M U L A T I O N : Logistic Regression
model1 = "LR"
classifier1 = LogisticRegression(random_state=0,max_iter=50)
hyperparameters1 = "max_iter=50"

model2 = "LR"
classifier2 = LogisticRegression(random_state=0,max_iter=150)
hyperparameters2 = "max_iter=150"

classifiers = [classifier1,classifier2]
models = [model1,model2]
hyperparameterss= [hyperparameters1,hyperparameters2]

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
            "/home/mireya/Documentos/Paper_3/Results_ICEdataset/results_%s.csv"%(model1),
            mode="a",
            newline="",
    ) as csvfile:
        # Create a CSV writer
        writer = csv.writer(csvfile)
        # Write the new row to the CSV file
        writer.writerow(new_row)

# #SECTION - S I M U L A T I O N : RANDOM FOREST
model1 = "RF"
classifier1 = RandomForestClassifier(random_state=0,max_depth=34,n_estimators=100)
hyperparameters1 = "max_depth,Nestimators=34,100"


model2 = "RF"
classifier2 = RandomForestClassifier(random_state=0,max_depth=1,n_estimators=15)
hyperparameters2 = "max_depth,Nestimators=1,15"


model3 = "RF"
classifier3 = RandomForestClassifier(random_state=0,max_depth=27,n_estimators=15)
hyperparameters3 ="max_depth,Nestimators=27,15"

model4 = "RF"
classifier4 = RandomForestClassifier(random_state=0,max_depth=11,n_estimators=20)
hyperparameters4 = "max_depth,Nestimators=11,20"


model5 = "RF"
classifier5 = RandomForestClassifier(random_state=0,max_depth=30,n_estimators=20)
hyperparameters5 = "max_depth,Nestimators=30,20"

classifiers = [classifier1, classifier2, classifier3, classifier4,classifier5]
models = [model1, model2, model3, model4,model5]
hyperparameterss= [hyperparameters1, hyperparameters2, hyperparameters3, hyperparameters4,hyperparameters5]

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
            "/home/mireya/Documentos/Paper_3/Results_ICEdataset/results_%s.csv"%(model1),
            mode="a",
            newline="",
    ) as csvfile:
        # Create a CSV writer
        writer = csv.writer(csvfile)
        # Write the new row to the CSV file
        writer.writerow(new_row)

#SECTION - S I M U L A T I O N : LAdaBoost
model1 = "Adaboost"
classifier1 = AdaBoostClassifier(random_state=0,n_estimators=200)
hyperparameters1 = "n_estimators=200"

model2 = "Adaboost"
classifier2 = AdaBoostClassifier(random_state=0,n_estimators=275)
hyperparameters2 = "max_iter=275"

classifiers = [classifier1,classifier2]
models = [model1,model2]
hyperparameterss= [hyperparameters1,hyperparameters2]


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
            "/home/mireya/Documentos/Paper_3/Results_ICEdataset/results_%s.csv"%(model1),
            mode="a",
            newline="",
    ) as csvfile:
        # Create a CSV writer
        writer = csv.writer(csvfile)
        # Write the new row to the CSV file
        writer.writerow(new_row)