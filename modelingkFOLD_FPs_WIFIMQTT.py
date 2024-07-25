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

dataset_path="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/Datasets_5s/dataset_binary.pkl"
with open(dataset_path,'rb') as f:
           dataset=pickle.load(f)


labels_path="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/Datasets_5s/labels_binary.pkl"
with open(labels_path,'rb') as f:
           labels=pickle.load(f)


data="FPs_WifiMQTT-Kfold10-Binary"
hash="Nilsimsa"
windowtime="5s"

# # #SECTION - S I M U L A T I O N 2: DECISION TREE

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

# classifiers = [classifier1]
# models = [model1]
# hyperparameterss= [hyperparameters1]

# model2 = "RF"
# classifier2 = RandomForestClassifier(random_state=0,max_depth=10,n_estimators=175)
# hyperparameters2 = "max_depth,Nestimators=10,175"
# classifiers = [classifier2]
# models = [model1]
# hyperparameterss= [hyperparameters2]

# model3 = "RF"
# classifier3 = RandomForestClassifier(random_state=0,max_depth=32,n_estimators=25)
# hyperparameters3 ="max_depth,Nestimators=32,25"
# classifiers = [classifier3]
# models = [model1]
# hyperparameterss= [hyperparameters3]

# model4 = "RF"
# classifier4 = RandomForestClassifier(random_state=0,max_depth=37,n_estimators=225)
# hyperparameters4 = "max_depth,Nestimators=37,225"
# classifiers = [classifier4]
# models = [model1]
# hyperparameterss= [hyperparameters4]


# model5 = "RF"
# classifier5 = RandomForestClassifier(random_state=0,max_depth=12,n_estimators=250)
# hyperparameters5 = "max_depth,Nestimators=12,250"
# classifiers = [classifier5]
# models = [model1]
# hyperparameterss= [hyperparameters5]


# #SECTION - S I M U L A T I O N 4: Logistic Regression
model1 = "LR"
classifier1 = LogisticRegression(random_state=0,max_iter=160)
hyperparameters1 = "max_iter=160"
classifiers = [classifier1]
models = [model1]
hyperparameterss= [hyperparameters1]

#SECTION - S I M U L A T I O N 5:
# model1 = "Adaboost"
# classifier1 = AdaBoostClassifier(random_state=0,n_estimators=125)
# hyperparameters1 = "Nestimators=125"

# model2 = "Adaboost"
# classifier2 = AdaBoostClassifier(random_state=0,n_estimators=150)
# hyperparameters2 = "Nestimators=150"

# model3 = "Adaboost"
# classifier3 = AdaBoostClassifier(random_state=0,n_estimators=225)
# hyperparameters3 = "Nestimators=225"

# classifiers = [classifier3]
# models = [model3]
# hyperparameterss= [hyperparameters3]

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
        CVResults["test_accu"].mean()*100,
        CVResults["test_prec"].mean()*100,
        CVResults["test_rec"].mean()*100,
        CVResults["test_f1score"].mean()*100,
        CVResults["test_auc_score"].mean()*100,
        CVResults["test_false_positive_rate"].mean()*100,
        CVResults["fit_time"],
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
