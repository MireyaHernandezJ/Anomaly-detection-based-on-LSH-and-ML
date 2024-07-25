
import csv
import pickle
import time
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import OneClassSVM
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier



#SECTION - FUNCTIONS to evaluate classifiers with kfold cross validation
def false_positive_rate(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # false positive
    fp = ((y_pred == 1) & (y_true == 0)).sum()

    # true negative
    tn = ((y_pred == 0) & (y_true == 0)).sum()

    # false positive rate
    return fp / (fp + tn)

def tp_value(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] and y_true[i] == 1:
            tp += 1
    return tp

def fp_value(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    fp=0
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i] and y_true[i] != 1:
            fp+=1
    return fp

def tn_value(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tn=0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] and y_true[i] != 1:
            tn+=1
    return tn

def fn_value(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    fn=0
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i] and y_true[i] == 1:
            fn+=1
    return fn
    
        
def aucScore(y_true, y_pred):
    auc_score=roc_auc_score(y_true,y_pred)
    return auc_score


def my_kfold_cross_validation(kfold_value, classifier, dataset, labels_binary):
    scoring = {
        "accu": make_scorer(accuracy_score),
        "prec": make_scorer(precision_score, pos_label=1),
        "rec": make_scorer(recall_score, pos_label=1),
        "f1score": make_scorer(f1_score, pos_label=1),
        "false_positive_rate": make_scorer(false_positive_rate),
        "tpv":make_scorer(tp_value),
        "fpv":make_scorer(fp_value),
        "tnv":make_scorer(tn_value),
        "fnv":make_scorer(fn_value),
        "auc_score":make_scorer(aucScore),
    }
    cv = KFold(n_splits=kfold_value, shuffle=True, random_state=42)
    cv_results = cross_validate(classifier, dataset, labels_binary, cv=cv, scoring=scoring)
    return cv_results

#DATASET:

data="FPS_ICE-kfold10-Binary"
hash="nilsimsa" #tlsh or nilsimsa
windowtime="5"#["5s","10s","15s","20s"]

dataset_path="/home/mireya/Documentos/Datasets-IoT/ICEdataset/ICEdataset_FPs_binary/dataset_%ss_nilsimsa.pkl"%(windowtime)
with open(dataset_path,'rb') as f:
           dataset=pickle.load(f)

labels_path="/home/mireya/Documentos/Datasets-IoT/ICEdataset/ICEdataset_FPs_binary/labels_%ss_nilsimsa.pkl"%(windowtime)
with open(labels_path,'rb') as f:
           labels=pickle.load(f)


# #SECTION - S I M U L A T I O N 2: DECISION TREE 5s
#5s:19,4,40,12,9
#10s:36,6,28,40,4
#15s:5,34,33,8,5
#20s:27,18,13,32,4
model1 = "DT"
classifier1 = DecisionTreeClassifier(random_state=0,max_depth=36)
hyperparameters1 = "max_depth=19"


model2 = "DT"
classifier2 = DecisionTreeClassifier(random_state=0,max_depth=6)
hyperparameters2 = "max_depth=4"


model3 = "DT"
classifier3 = DecisionTreeClassifier(random_state=0,max_depth=28)
hyperparameters3 = "max_depth=40"

model4 = "DT"
classifier4 = DecisionTreeClassifier(random_state=0,max_depth=40)
hyperparameters4 = "max_depth=12"

model5 = "DT"
classifier5 = DecisionTreeClassifier(random_state=0,max_depth=4)
hyperparameters5 = "max_depth=9"

classifiers = [classifier1, classifier2, classifier3, classifier4,classifier5]
models = [model1, model2, model3, model4,model5]
hyperparameterss= [hyperparameters1, hyperparameters2, hyperparameters3, hyperparameters4,hyperparameters5]


################## S A V I N G

for idx, classifier in enumerate(classifiers):

    CVResults=my_kfold_cross_validation(kfold_value=10, classifier=classifier, dataset=dataset, labels_binary=labels)
    attack_train_samples=""
    normal_train_samples=""
    attack_test_samples=""
    normal_test_samples=""
    samples_train=""
    samples_test=""

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
        CVResults["test_tnv"],
        CVResults["test_fpv"],
        CVResults["test_fnv"],
        CVResults["test_tpv"],
        CVResults["test_accu"]*100,
        CVResults["test_accu"].mean()*100,
        CVResults["test_prec"]*100,
        CVResults["test_prec"].mean()*100,
        CVResults["test_rec"]*100,
        CVResults["test_rec"].mean()*100,
        CVResults["test_f1score"]*100,
        CVResults["test_f1score"].mean()*100,
        CVResults["test_auc_score"]*100,
        CVResults["test_auc_score"].mean()*100,
        CVResults["test_false_positive_rate"]*100,
        CVResults["test_false_positive_rate"].mean()*100,
        CVResults["fit_time"],
        CVResults["fit_time"].mean(),
    ]

    # Open the CSV file in append mode
    with open(
            "/home/mireya/Documentos/Paper_3/Results_ICEdataset/resultskfold_%s.csv"%(model1),
            mode="a",
            newline="",
    ) as csvfile:
        # Create a CSV writer
        writer = csv.writer(csvfile)
        # Write the new row to the CSV file
        writer.writerow(new_row)

#SECTION - S I M U L A T I O N : Logistic Regression 5s
#5s:180,125
#10s:50,150
#15s:25,50,
#20s:50,25
model1 = "LR"
classifier1 = LogisticRegression(random_state=0,max_iter=180)
hyperparameters1 = "max_iter=180"

model2 = "LR"
classifier2 = LogisticRegression(random_state=0,max_iter=125)
hyperparameters2 = "max_iter=125"

classifiers = [classifier1,classifier2]
models = [model1,model2]
hyperparameterss= [hyperparameters1,hyperparameters2]


################## S A V I N G

for idx, classifier in enumerate(classifiers):

    CVResults=my_kfold_cross_validation(kfold_value=10, classifier=classifier, dataset=dataset, labels_binary=labels)
    attack_train_samples=""
    normal_train_samples=""
    attack_test_samples=""
    normal_test_samples=""
    samples_train=""
    samples_test=""

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
        CVResults["test_tnv"],
        CVResults["test_fpv"],
        CVResults["test_fnv"],
        CVResults["test_tpv"],
        CVResults["test_accu"]*100,
        CVResults["test_accu"].mean()*100,
        CVResults["test_prec"]*100,
        CVResults["test_prec"].mean()*100,
        CVResults["test_rec"]*100,
        CVResults["test_rec"].mean()*100,
        CVResults["test_f1score"]*100,
        CVResults["test_f1score"].mean()*100,
        CVResults["test_auc_score"]*100,
        CVResults["test_auc_score"].mean()*100,
        CVResults["test_false_positive_rate"]*100,
        CVResults["test_false_positive_rate"].mean()*100,
        CVResults["fit_time"],
        CVResults["fit_time"].mean(),
    ]

    # Open the CSV file in append mode
    with open(
            "/home/mireya/Documentos/Paper_3/Results_ICEdataset/resultskfold_%s.csv"%(model1),
            mode="a",
            newline="",
    ) as csvfile:
        # Create a CSV writer
        writer = csv.writer(csvfile)
        # Write the new row to the CSV file
        writer.writerow(new_row)

# #SECTION - S I M U L A T I O N : RANDOM FOREST 5s
#5s:17-287,1-125,27-15,25-200,10-175,
#10s:34-100,1-15,27-15,11-20,30-20,
#15s:15-25,1-5,18-15,37-5,31-25,
#20s:15-225,1-10,14-275,18-175,9-150
model1 = "RF"
classifier1 = RandomForestClassifier(random_state=0,max_depth=17,n_estimators=287)
hyperparameters1 = "max_depth,Nestimators=17,287"


model2 = "RF"
classifier2 = RandomForestClassifier(random_state=0,max_depth=1,n_estimators=125)
hyperparameters2 = "max_depth,Nestimators=1,125"


model3 = "RF"
classifier3 = RandomForestClassifier(random_state=0,max_depth=27,n_estimators=15)
hyperparameters3 ="max_depth,Nestimators=27,15"

model4 = "RF"
classifier4 = RandomForestClassifier(random_state=0,max_depth=25,n_estimators=200)
hyperparameters4 = "max_depth,Nestimators=25,200"


model5 = "RF"
classifier5 = RandomForestClassifier(random_state=0,max_depth=10,n_estimators=175)
hyperparameters5 = "max_depth,Nestimators=10,175"

classifiers = [classifier1, classifier2, classifier3, classifier4,classifier5]
models = [model1, model2, model3, model4,model5]
hyperparameterss= [hyperparameters1, hyperparameters2, hyperparameters3, hyperparameters4,hyperparameters5]

################## S A V I N G

for idx, classifier in enumerate(classifiers):

    CVResults=my_kfold_cross_validation(kfold_value=10, classifier=classifier, dataset=dataset, labels_binary=labels)
    attack_train_samples=""
    normal_train_samples=""
    attack_test_samples=""
    normal_test_samples=""
    samples_train=""
    samples_test=""

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
        CVResults["test_tnv"],
        CVResults["test_fpv"],
        CVResults["test_fnv"],
        CVResults["test_tpv"],
        CVResults["test_accu"]*100,
        CVResults["test_accu"].mean()*100,
        CVResults["test_prec"]*100,
        CVResults["test_prec"].mean()*100,
        CVResults["test_rec"]*100,
        CVResults["test_rec"].mean()*100,
        CVResults["test_f1score"]*100,
        CVResults["test_f1score"].mean()*100,
        CVResults["test_auc_score"]*100,
        CVResults["test_auc_score"].mean()*100,
        CVResults["test_false_positive_rate"]*100,
        CVResults["test_false_positive_rate"].mean()*100,
        CVResults["fit_time"],
        CVResults["fit_time"].mean(),
    ]

    # Open the CSV file in append mode
    with open(
            "/home/mireya/Documentos/Paper_3/Results_ICEdataset/resultskfold_%s.csv"%(model1),
            mode="a",
            newline="",
    ) as csvfile:
        # Create a CSV writer
        writer = csv.writer(csvfile)
        # Write the new row to the CSV file
        writer.writerow(new_row)

#SECTION - S I M U L A T I O N : AdaBoost 5s
#5s:125,50
#10s:200,275
#15s:200,275
#20s:50,100,25
model1 = "Adaboost"
classifier1 = AdaBoostClassifier(random_state=0,n_estimators=125)
hyperparameters1 = "n_estimators=125"

model2 = "Adaboost"
classifier2 = AdaBoostClassifier(random_state=0,n_estimators=50)
hyperparameters2 = "max_iter=50"

classifiers = [classifier1,classifier2]
models = [model1,model2]
hyperparameterss= [hyperparameters1,hyperparameters2]


################## S A V I N G

for idx, classifier in enumerate(classifiers):

    CVResults=my_kfold_cross_validation(kfold_value=10, classifier=classifier, dataset=dataset, labels_binary=labels)
    attack_train_samples=""
    normal_train_samples=""
    attack_test_samples=""
    normal_test_samples=""
    samples_train=""
    samples_test=""

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
        CVResults["test_tnv"],
        CVResults["test_fpv"],
        CVResults["test_fnv"],
        CVResults["test_tpv"],
        CVResults["test_accu"]*100,
        CVResults["test_accu"].mean()*100,
        CVResults["test_prec"]*100,
        CVResults["test_prec"].mean()*100,
        CVResults["test_rec"]*100,
        CVResults["test_rec"].mean()*100,
        CVResults["test_f1score"]*100,
        CVResults["test_f1score"].mean()*100,
        CVResults["test_auc_score"]*100,
        CVResults["test_auc_score"].mean()*100,
        CVResults["test_false_positive_rate"]*100,
        CVResults["test_false_positive_rate"].mean()*100,
        CVResults["fit_time"],
        CVResults["fit_time"].mean(),
    ]

    # Open the CSV file in append mode
    with open(
            "/home/mireya/Documentos/Paper_3/Results_ICEdataset/resultskfold_%s.csv"%(model1),
            mode="a",
            newline="",
    ) as csvfile:
        # Create a CSV writer
        writer = csv.writer(csvfile)
        # Write the new row to the CSV file
        writer.writerow(new_row)

# #SECTION - S I M U L A T I O N 2: DECISION TREE 10s
#5s:19,4,40,12,9
#10s:36,6,28,40,4
#15s:5,34,33,8,5
#20s:27,18,13,32,4
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

    CVResults=my_kfold_cross_validation(kfold_value=10, classifier=classifier, dataset=dataset, labels_binary=labels)
    attack_train_samples=""
    normal_train_samples=""
    attack_test_samples=""
    normal_test_samples=""
    samples_train=""
    samples_test=""

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
        CVResults["test_tnv"],
        CVResults["test_fpv"],
        CVResults["test_fnv"],
        CVResults["test_tpv"],
        CVResults["test_accu"]*100,
        CVResults["test_accu"].mean()*100,
        CVResults["test_prec"]*100,
        CVResults["test_prec"].mean()*100,
        CVResults["test_rec"]*100,
        CVResults["test_rec"].mean()*100,
        CVResults["test_f1score"]*100,
        CVResults["test_f1score"].mean()*100,
        CVResults["test_auc_score"]*100,
        CVResults["test_auc_score"].mean()*100,
        CVResults["test_false_positive_rate"]*100,
        CVResults["test_false_positive_rate"].mean()*100,
        CVResults["fit_time"],
        CVResults["fit_time"].mean(),
    ]

    # Open the CSV file in append mode
    with open(
            "/home/mireya/Documentos/Paper_3/Results_ICEdataset/resultskfold_%s.csv"%(model1),
            mode="a",
            newline="",
    ) as csvfile:
        # Create a CSV writer
        writer = csv.writer(csvfile)
        # Write the new row to the CSV file
        writer.writerow(new_row)


#SECTION - S I M U L A T I O N : Logistic Regression 10s
#5s:180,125
#10s:50,150
#15s:25,50,
#20s:50,25
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

    CVResults=my_kfold_cross_validation(kfold_value=10, classifier=classifier, dataset=dataset, labels_binary=labels)
    attack_train_samples=""
    normal_train_samples=""
    attack_test_samples=""
    normal_test_samples=""
    samples_train=""
    samples_test=""

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
        CVResults["test_tnv"],
        CVResults["test_fpv"],
        CVResults["test_fnv"],
        CVResults["test_tpv"],
        CVResults["test_accu"]*100,
        CVResults["test_accu"].mean()*100,
        CVResults["test_prec"]*100,
        CVResults["test_prec"].mean()*100,
        CVResults["test_rec"]*100,
        CVResults["test_rec"].mean()*100,
        CVResults["test_f1score"]*100,
        CVResults["test_f1score"].mean()*100,
        CVResults["test_auc_score"]*100,
        CVResults["test_auc_score"].mean()*100,
        CVResults["test_false_positive_rate"]*100,
        CVResults["test_false_positive_rate"].mean()*100,
        CVResults["fit_time"],
        CVResults["fit_time"].mean(),
    ]

    # Open the CSV file in append mode
    with open(
            "/home/mireya/Documentos/Paper_3/Results_ICEdataset/resultskfold_%s.csv"%(model1),
            mode="a",
            newline="",
    ) as csvfile:
        # Create a CSV writer
        writer = csv.writer(csvfile)
        # Write the new row to the CSV file
        writer.writerow(new_row)


# #SECTION - S I M U L A T I O N : RANDOM FOREST 10s
#5s:17-287,1-125,27-15,25-200,10-175,
#10s:34-100,1-15,27-15,11-20,30-20,
#15s:15-25,1-5,18-15,37-5,31-25,
#20s:15-225,1-10,14-275,18-175,9-150
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

    CVResults=my_kfold_cross_validation(kfold_value=10, classifier=classifier, dataset=dataset, labels_binary=labels)
    attack_train_samples=""
    normal_train_samples=""
    attack_test_samples=""
    normal_test_samples=""
    samples_train=""
    samples_test=""

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
        CVResults["test_tnv"],
        CVResults["test_fpv"],
        CVResults["test_fnv"],
        CVResults["test_tpv"],
        CVResults["test_accu"]*100,
        CVResults["test_accu"].mean()*100,
        CVResults["test_prec"]*100,
        CVResults["test_prec"].mean()*100,
        CVResults["test_rec"]*100,
        CVResults["test_rec"].mean()*100,
        CVResults["test_f1score"]*100,
        CVResults["test_f1score"].mean()*100,
        CVResults["test_auc_score"]*100,
        CVResults["test_auc_score"].mean()*100,
        CVResults["test_false_positive_rate"]*100,
        CVResults["test_false_positive_rate"].mean()*100,
        CVResults["fit_time"],
        CVResults["fit_time"].mean(),
    ]

    # Open the CSV file in append mode
    with open(
            "/home/mireya/Documentos/Paper_3/Results_ICEdataset/resultskfold_%s.csv"%(model1),
            mode="a",
            newline="",
    ) as csvfile:
        # Create a CSV writer
        writer = csv.writer(csvfile)
        # Write the new row to the CSV file
        writer.writerow(new_row)

#SECTION - S I M U L A T I O N : AdaBoost 10s
#5s:125,50
#10s:200,275
#15s:200,275
#20s:50,100,25
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

    CVResults=my_kfold_cross_validation(kfold_value=10, classifier=classifier, dataset=dataset, labels_binary=labels)
    attack_train_samples=""
    normal_train_samples=""
    attack_test_samples=""
    normal_test_samples=""
    samples_train=""
    samples_test=""

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
        CVResults["test_tnv"],
        CVResults["test_fpv"],
        CVResults["test_fnv"],
        CVResults["test_tpv"],
        CVResults["test_accu"]*100,
        CVResults["test_accu"].mean()*100,
        CVResults["test_prec"]*100,
        CVResults["test_prec"].mean()*100,
        CVResults["test_rec"]*100,
        CVResults["test_rec"].mean()*100,
        CVResults["test_f1score"]*100,
        CVResults["test_f1score"].mean()*100,
        CVResults["test_auc_score"]*100,
        CVResults["test_auc_score"].mean()*100,
        CVResults["test_false_positive_rate"]*100,
        CVResults["test_false_positive_rate"].mean()*100,
        CVResults["fit_time"],
        CVResults["fit_time"].mean(),
    ]

    # Open the CSV file in append mode
    with open(
            "/home/mireya/Documentos/Paper_3/Results_ICEdataset/resultskfold_%s.csv"%(model1),
            mode="a",
            newline="",
    ) as csvfile:
        # Create a CSV writer
        writer = csv.writer(csvfile)
        # Write the new row to the CSV file
        writer.writerow(new_row)

# #SECTION - S I M U L A T I O N 2: DECISION TREE 15s
#5s:19,4,40,12,9
#10s:36,6,28,40,4
#15s:5,34,33,8,5
#20s:27,18,13,32,4
model1 = "DT"
classifier1 = DecisionTreeClassifier(random_state=0,max_depth=36)
hyperparameters1 = "max_depth=5"


model2 = "DT"
classifier2 = DecisionTreeClassifier(random_state=0,max_depth=6)
hyperparameters2 = "max_depth=34"


model3 = "DT"
classifier3 = DecisionTreeClassifier(random_state=0,max_depth=28)
hyperparameters3 = "max_depth=33"

model4 = "DT"
classifier4 = DecisionTreeClassifier(random_state=0,max_depth=40)
hyperparameters4 = "max_depth=8"

model5 = "DT"
classifier5 = DecisionTreeClassifier(random_state=0,max_depth=4)
hyperparameters5 = "max_depth=5"

classifiers = [classifier1, classifier2, classifier3, classifier4,classifier5]
models = [model1, model2, model3, model4,model5]
hyperparameterss= [hyperparameters1, hyperparameters2, hyperparameters3, hyperparameters4,hyperparameters5]


################## S A V I N G

for idx, classifier in enumerate(classifiers):

    CVResults=my_kfold_cross_validation(kfold_value=10, classifier=classifier, dataset=dataset, labels_binary=labels)
    attack_train_samples=""
    normal_train_samples=""
    attack_test_samples=""
    normal_test_samples=""
    samples_train=""
    samples_test=""

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
        CVResults["test_tnv"],
        CVResults["test_fpv"],
        CVResults["test_fnv"],
        CVResults["test_tpv"],
        CVResults["test_accu"]*100,
        CVResults["test_accu"].mean()*100,
        CVResults["test_prec"]*100,
        CVResults["test_prec"].mean()*100,
        CVResults["test_rec"]*100,
        CVResults["test_rec"].mean()*100,
        CVResults["test_f1score"]*100,
        CVResults["test_f1score"].mean()*100,
        CVResults["test_auc_score"]*100,
        CVResults["test_auc_score"].mean()*100,
        CVResults["test_false_positive_rate"]*100,
        CVResults["test_false_positive_rate"].mean()*100,
        CVResults["fit_time"],
        CVResults["fit_time"].mean(),
    ]

    # Open the CSV file in append mode
    with open(
            "/home/mireya/Documentos/Paper_3/Results_ICEdataset/resultskfold_%s.csv"%(model1),
            mode="a",
            newline="",
    ) as csvfile:
        # Create a CSV writer
        writer = csv.writer(csvfile)
        # Write the new row to the CSV file
        writer.writerow(new_row)

#SECTION - S I M U L A T I O N : Logistic Regression 15s
#5s:180,125
#10s:50,150
#15s:25,50,
#20s:50,25
model1 = "LR"
classifier1 = LogisticRegression(random_state=0,max_iter=50)
hyperparameters1 = "max_iter=50"

model2 = "LR"
classifier2 = LogisticRegression(random_state=0,max_iter=25)
hyperparameters2 = "max_iter=25"

classifiers = [classifier1,classifier2]
models = [model1,model2]
hyperparameterss= [hyperparameters1,hyperparameters2]



################## S A V I N G

for idx, classifier in enumerate(classifiers):

    CVResults=my_kfold_cross_validation(kfold_value=10, classifier=classifier, dataset=dataset, labels_binary=labels)
    attack_train_samples=""
    normal_train_samples=""
    attack_test_samples=""
    normal_test_samples=""
    samples_train=""
    samples_test=""

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
        CVResults["test_tnv"],
        CVResults["test_fpv"],
        CVResults["test_fnv"],
        CVResults["test_tpv"],
        CVResults["test_accu"]*100,
        CVResults["test_accu"].mean()*100,
        CVResults["test_prec"]*100,
        CVResults["test_prec"].mean()*100,
        CVResults["test_rec"]*100,
        CVResults["test_rec"].mean()*100,
        CVResults["test_f1score"]*100,
        CVResults["test_f1score"].mean()*100,
        CVResults["test_auc_score"]*100,
        CVResults["test_auc_score"].mean()*100,
        CVResults["test_false_positive_rate"]*100,
        CVResults["test_false_positive_rate"].mean()*100,
        CVResults["fit_time"],
        CVResults["fit_time"].mean(),
    ]

    # Open the CSV file in append mode
    with open(
            "/home/mireya/Documentos/Paper_3/Results_ICEdataset/resultskfold_%s.csv"%(model1),
            mode="a",
            newline="",
    ) as csvfile:
        # Create a CSV writer
        writer = csv.writer(csvfile)
        # Write the new row to the CSV file
        writer.writerow(new_row)

# #SECTION - S I M U L A T I O N : RANDOM FOREST 15s
#15s:15-25,1-5,18-15,37-5,31-25,
#20s:15-225,1-10,14-275,18-175,9-150
model1 = "RF"
classifier1 = RandomForestClassifier(random_state=0,max_depth=15,n_estimators=25)
hyperparameters1 = "max_depth,Nestimators=15,25"


model2 = "RF"
classifier2 = RandomForestClassifier(random_state=0,max_depth=1,n_estimators=5)
hyperparameters2 = "max_depth,Nestimators=1,5"


model3 = "RF"
classifier3 = RandomForestClassifier(random_state=0,max_depth=18,n_estimators=15)
hyperparameters3 ="max_depth,Nestimators=18,15"

model4 = "RF"
classifier4 = RandomForestClassifier(random_state=0,max_depth=37,n_estimators=5)
hyperparameters4 = "max_depth,Nestimators=37,5"


model5 = "RF"
classifier5 = RandomForestClassifier(random_state=0,max_depth=31,n_estimators=25)
hyperparameters5 = "max_depth,Nestimators=31,25"

classifiers = [classifier1, classifier2, classifier3, classifier4,classifier5]
models = [model1, model2, model3, model4,model5]
hyperparameterss= [hyperparameters1, hyperparameters2, hyperparameters3, hyperparameters4,hyperparameters5]

################## S A V I N G

for idx, classifier in enumerate(classifiers):

    CVResults=my_kfold_cross_validation(kfold_value=10, classifier=classifier, dataset=dataset, labels_binary=labels)
    attack_train_samples=""
    normal_train_samples=""
    attack_test_samples=""
    normal_test_samples=""
    samples_train=""
    samples_test=""

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
        CVResults["test_tnv"],
        CVResults["test_fpv"],
        CVResults["test_fnv"],
        CVResults["test_tpv"],
        CVResults["test_accu"]*100,
        CVResults["test_accu"].mean()*100,
        CVResults["test_prec"]*100,
        CVResults["test_prec"].mean()*100,
        CVResults["test_rec"]*100,
        CVResults["test_rec"].mean()*100,
        CVResults["test_f1score"]*100,
        CVResults["test_f1score"].mean()*100,
        CVResults["test_auc_score"]*100,
        CVResults["test_auc_score"].mean()*100,
        CVResults["test_false_positive_rate"]*100,
        CVResults["test_false_positive_rate"].mean()*100,
        CVResults["fit_time"],
        CVResults["fit_time"].mean(),
    ]

    # Open the CSV file in append mode
    with open(
            "/home/mireya/Documentos/Paper_3/Results_ICEdataset/resultskfold_%s.csv"%(model1),
            mode="a",
            newline="",
    ) as csvfile:
        # Create a CSV writer
        writer = csv.writer(csvfile)
        # Write the new row to the CSV file
        writer.writerow(new_row)

#SECTION - S I M U L A T I O N : AdaBoost 15s
#5s:125,50
#10s:200,275
#15s:200,275
#20s:50,100,25
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

    CVResults=my_kfold_cross_validation(kfold_value=10, classifier=classifier, dataset=dataset, labels_binary=labels)
    attack_train_samples=""
    normal_train_samples=""
    attack_test_samples=""
    normal_test_samples=""
    samples_train=""
    samples_test=""

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
        CVResults["test_tnv"],
        CVResults["test_fpv"],
        CVResults["test_fnv"],
        CVResults["test_tpv"],
        CVResults["test_accu"]*100,
        CVResults["test_accu"].mean()*100,
        CVResults["test_prec"]*100,
        CVResults["test_prec"].mean()*100,
        CVResults["test_rec"]*100,
        CVResults["test_rec"].mean()*100,
        CVResults["test_f1score"]*100,
        CVResults["test_f1score"].mean()*100,
        CVResults["test_auc_score"]*100,
        CVResults["test_auc_score"].mean()*100,
        CVResults["test_false_positive_rate"]*100,
        CVResults["test_false_positive_rate"].mean()*100,
        CVResults["fit_time"],
        CVResults["fit_time"].mean(),
    ]

    # Open the CSV file in append mode
    with open(
            "/home/mireya/Documentos/Paper_3/Results_ICEdataset/resultskfold_%s.csv"%(model1),
            mode="a",
            newline="",
    ) as csvfile:
        # Create a CSV writer
        writer = csv.writer(csvfile)
        # Write the new row to the CSV file
        writer.writerow(new_row)

# #SECTION - S I M U L A T I O N 2: DECISION TREE 20
#5s:19,4,40,12,9
#10s:36,6,28,40,4
#15s:5,34,33,8,5
#20s:27,18,13,32,4
model1 = "DT"
classifier1 = DecisionTreeClassifier(random_state=0,max_depth=36)
hyperparameters1 = "max_depth=27"


model2 = "DT"
classifier2 = DecisionTreeClassifier(random_state=0,max_depth=6)
hyperparameters2 = "max_depth=18"


model3 = "DT"
classifier3 = DecisionTreeClassifier(random_state=0,max_depth=28)
hyperparameters3 = "max_depth=13"

model4 = "DT"
classifier4 = DecisionTreeClassifier(random_state=0,max_depth=40)
hyperparameters4 = "max_depth=32"

model5 = "DT"
classifier5 = DecisionTreeClassifier(random_state=0,max_depth=4)
hyperparameters5 = "max_depth=4"

classifiers = [classifier1, classifier2, classifier3, classifier4,classifier5]
models = [model1, model2, model3, model4,model5]
hyperparameterss= [hyperparameters1, hyperparameters2, hyperparameters3, hyperparameters4,hyperparameters5]

################## S A V I N G

for idx, classifier in enumerate(classifiers):

    CVResults=my_kfold_cross_validation(kfold_value=10, classifier=classifier, dataset=dataset, labels_binary=labels)
    attack_train_samples=""
    normal_train_samples=""
    attack_test_samples=""
    normal_test_samples=""
    samples_train=""
    samples_test=""

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
        CVResults["test_tnv"],
        CVResults["test_fpv"],
        CVResults["test_fnv"],
        CVResults["test_tpv"],
        CVResults["test_accu"]*100,
        CVResults["test_accu"].mean()*100,
        CVResults["test_prec"]*100,
        CVResults["test_prec"].mean()*100,
        CVResults["test_rec"]*100,
        CVResults["test_rec"].mean()*100,
        CVResults["test_f1score"]*100,
        CVResults["test_f1score"].mean()*100,
        CVResults["test_auc_score"]*100,
        CVResults["test_auc_score"].mean()*100,
        CVResults["test_false_positive_rate"]*100,
        CVResults["test_false_positive_rate"].mean()*100,
        CVResults["fit_time"],
        CVResults["fit_time"].mean(),
    ]

    # Open the CSV file in append mode
    with open(
            "/home/mireya/Documentos/Paper_3/Results_ICEdataset/resultskfold_%s.csv"%(model1),
            mode="a",
            newline="",
    ) as csvfile:
        # Create a CSV writer
        writer = csv.writer(csvfile)
        # Write the new row to the CSV file
        writer.writerow(new_row)

#SECTION - S I M U L A T I O N : Logistic Regression 20s
#5s:180,125
#10s:50,150
#15s:25,50,
#20s:50,25
model1 = "LR"
classifier1 = LogisticRegression(random_state=0,max_iter=50)
hyperparameters1 = "max_iter=50"

model2 = "LR"
classifier2 = LogisticRegression(random_state=0,max_iter=25)
hyperparameters2 = "max_iter=25"

classifiers = [classifier1,classifier2]
models = [model1,model2]
hyperparameterss= [hyperparameters1,hyperparameters2]


################## S A V I N G

for idx, classifier in enumerate(classifiers):

    CVResults=my_kfold_cross_validation(kfold_value=10, classifier=classifier, dataset=dataset, labels_binary=labels)
    attack_train_samples=""
    normal_train_samples=""
    attack_test_samples=""
    normal_test_samples=""
    samples_train=""
    samples_test=""

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
        CVResults["test_tnv"],
        CVResults["test_fpv"],
        CVResults["test_fnv"],
        CVResults["test_tpv"],
        CVResults["test_accu"]*100,
        CVResults["test_accu"].mean()*100,
        CVResults["test_prec"]*100,
        CVResults["test_prec"].mean()*100,
        CVResults["test_rec"]*100,
        CVResults["test_rec"].mean()*100,
        CVResults["test_f1score"]*100,
        CVResults["test_f1score"].mean()*100,
        CVResults["test_auc_score"]*100,
        CVResults["test_auc_score"].mean()*100,
        CVResults["test_false_positive_rate"]*100,
        CVResults["test_false_positive_rate"].mean()*100,
        CVResults["fit_time"],
        CVResults["fit_time"].mean(),
    ]

    # Open the CSV file in append mode
    with open(
            "/home/mireya/Documentos/Paper_3/Results_ICEdataset/resultskfold_%s.csv"%(model1),
            mode="a",
            newline="",
    ) as csvfile:
        # Create a CSV writer
        writer = csv.writer(csvfile)
        # Write the new row to the CSV file
        writer.writerow(new_row)

# #SECTION - S I M U L A T I O N : RANDOM FOREST 20s
#20s:15-225,1-10,14-275,18-175,9-150
model1 = "RF"
classifier1 = RandomForestClassifier(random_state=0,max_depth=15,n_estimators=225)
hyperparameters1 = "max_depth,Nestimators=15,225"


model2 = "RF"
classifier2 = RandomForestClassifier(random_state=0,max_depth=1,n_estimators=10)
hyperparameters2 = "max_depth,Nestimators=1,10"


model3 = "RF"
classifier3 = RandomForestClassifier(random_state=0,max_depth=14,n_estimators=275)
hyperparameters3 ="max_depth,Nestimators=14,275"

model4 = "RF"
classifier4 = RandomForestClassifier(random_state=0,max_depth=18,n_estimators=175)
hyperparameters4 = "max_depth,Nestimators=18,175"


model5 = "RF"
classifier5 = RandomForestClassifier(random_state=0,max_depth=9,n_estimators=150)
hyperparameters5 = "max_depth,Nestimators=9,150"

classifiers = [classifier1, classifier2, classifier3, classifier4,classifier5]
models = [model1, model2, model3, model4,model5]
hyperparameterss= [hyperparameters1, hyperparameters2, hyperparameters3, hyperparameters4,hyperparameters5]


################## S A V I N G

for idx, classifier in enumerate(classifiers):

    CVResults=my_kfold_cross_validation(kfold_value=10, classifier=classifier, dataset=dataset, labels_binary=labels)
    attack_train_samples=""
    normal_train_samples=""
    attack_test_samples=""
    normal_test_samples=""
    samples_train=""
    samples_test=""

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
        CVResults["test_tnv"],
        CVResults["test_fpv"],
        CVResults["test_fnv"],
        CVResults["test_tpv"],
        CVResults["test_accu"]*100,
        CVResults["test_accu"].mean()*100,
        CVResults["test_prec"]*100,
        CVResults["test_prec"].mean()*100,
        CVResults["test_rec"]*100,
        CVResults["test_rec"].mean()*100,
        CVResults["test_f1score"]*100,
        CVResults["test_f1score"].mean()*100,
        CVResults["test_auc_score"]*100,
        CVResults["test_auc_score"].mean()*100,
        CVResults["test_false_positive_rate"]*100,
        CVResults["test_false_positive_rate"].mean()*100,
        CVResults["fit_time"],
        CVResults["fit_time"].mean(),
    ]

    # Open the CSV file in append mode
    with open(
            "/home/mireya/Documentos/Paper_3/Results_ICEdataset/resultskfold_%s.csv"%(model1),
            mode="a",
            newline="",
    ) as csvfile:
        # Create a CSV writer
        writer = csv.writer(csvfile)
        # Write the new row to the CSV file
        writer.writerow(new_row)

#SECTION - S I M U L A T I O N : AdaBoost 5s
#5s:125,50
#10s:200,275
#15s:200,275
#20s:50,100,25
model1 = "Adaboost"
classifier1 = AdaBoostClassifier(random_state=0,n_estimators=50)
hyperparameters1 = "n_estimators=50"

model2 = "Adaboost"
classifier2 = AdaBoostClassifier(random_state=0,n_estimators=100)
hyperparameters2 = "max_iter=100"

model3 = "Adaboost"
classifier3 = AdaBoostClassifier(random_state=0,n_estimators=25)
hyperparameters3 = "max_iter=25"

classifiers = [classifier1,classifier2,classifier3]
models = [model1,model2,model3]
hyperparameterss= [hyperparameters1,hyperparameters2,hyperparameters3]


################## S A V I N G

for idx, classifier in enumerate(classifiers):

    CVResults=my_kfold_cross_validation(kfold_value=10, classifier=classifier, dataset=dataset, labels_binary=labels)
    attack_train_samples=""
    normal_train_samples=""
    attack_test_samples=""
    normal_test_samples=""
    samples_train=""
    samples_test=""

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
        CVResults["test_tnv"],
        CVResults["test_fpv"],
        CVResults["test_fnv"],
        CVResults["test_tpv"],
        CVResults["test_accu"]*100,
        CVResults["test_accu"].mean()*100,
        CVResults["test_prec"]*100,
        CVResults["test_prec"].mean()*100,
        CVResults["test_rec"]*100,
        CVResults["test_rec"].mean()*100,
        CVResults["test_f1score"]*100,
        CVResults["test_f1score"].mean()*100,
        CVResults["test_auc_score"]*100,
        CVResults["test_auc_score"].mean()*100,
        CVResults["test_false_positive_rate"]*100,
        CVResults["test_false_positive_rate"].mean()*100,
        CVResults["fit_time"],
        CVResults["fit_time"].mean(),
    ]

    # Open the CSV file in append mode
    with open(
            "/home/mireya/Documentos/Paper_3/Results_ICEdataset/resultskfold_%s.csv"%(model1),
            mode="a",
            newline="",
    ) as csvfile:
        # Create a CSV writer
        writer = csv.writer(csvfile)
        # Write the new row to the CSV file
        writer.writerow(new_row)
