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
    roc_curve,
)
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import OneClassSVM
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
import matplotlib.pyplot as plt


def my_kfold_cross_validation(kfold_value, classifier, dataset, labels_binary):
    cv=KFold(n_splits=kfold_value,shuffle=True,random_state=42)
    y_scores=cross_val_predict(classifier,dataset,labels_binary,cv=cv,method='predict_proba')[:,1]
    threshold=0.5
    y_pred=np.where(y_scores>=threshold,1,0)
    return y_pred

def plot_roc_curve(labels_test,results_classifier,model,timewindow):
        fpr,tpr,_=roc_curve(labels_test,results_classifier)
        AUC_score=roc_auc_score(labels_test,results_classifier)
        AUC_score=AUC_score*100
        plt.plot(fpr,tpr,label=f'AUC_{model}={AUC_score:.2f}%')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title(f'ROC Curve within a time window of {timewindow} seconds')
        plt.legend()
        plt.savefig(f'roc_curve_{timewindow}.png')


# # #SECTION - S I M U L A T I O N 5s

# data="FPS_ICE-kfold10-Binary"
# hash="nilsimsa" 
# windowtime="5"

# dataset_path="/home/mireya/Documentos/Datasets-IoT/ICEdataset/ICEdataset_FPs_binary/dataset_%ss_nilsimsa.pkl"%(windowtime)
# with open(dataset_path,'rb') as f:
#            dataset=pickle.load(f)

# labels_path="/home/mireya/Documentos/Datasets-IoT/ICEdataset/ICEdataset_FPs_binary/labels_%ss_nilsimsa.pkl"%(windowtime)
# with open(labels_path,'rb') as f:
#            labels=pickle.load(f)

# classifier2 = DecisionTreeClassifier(random_state=0,max_depth=36)
# classifier1 = LogisticRegression(random_state=0,max_iter=125)
# classifier3 = RandomForestClassifier(random_state=0,max_depth=18,n_estimators=175)
# classifier4 = AdaBoostClassifier(random_state=0,n_estimators=275)
# model2='DT'
# model1='LR'
# model3='RF'
# model4='Adaboost'

# classifiers=[classifier1,classifier2,classifier3,classifier4]
# models=[model1,model2,model3,model4]

# for idx, classifier in enumerate(classifiers):

#     results=my_kfold_cross_validation(kfold_value=10, classifier=classifier, dataset=dataset, labels_binary=labels)
#     plot_roc_curve(labels_test=labels,results_classifier=results,model=models[idx],timewindow=windowtime)




# # #SECTION - S I M U L A T I O N 10

# data="FPS_ICE-kfold10-Binary"
# hash="nilsimsa" 
# windowtime="10"

# dataset_path="/home/mireya/Documentos/Datasets-IoT/ICEdataset/ICEdataset_FPs_binary/dataset_%ss_nilsimsa.pkl"%(windowtime)
# with open(dataset_path,'rb') as f:
#            dataset=pickle.load(f)

# labels_path="/home/mireya/Documentos/Datasets-IoT/ICEdataset/ICEdataset_FPs_binary/labels_%ss_nilsimsa.pkl"%(windowtime)
# with open(labels_path,'rb') as f:
#            labels=pickle.load(f)

# classifier2 = DecisionTreeClassifier(random_state=0,max_depth=34)
# classifier1 = LogisticRegression(random_state=0,max_iter=50)
# classifier3 = RandomForestClassifier(random_state=0,max_depth=17,n_estimators=287)
# classifier4 = AdaBoostClassifier(random_state=0,n_estimators=125)
# model2='DT'
# model1='LR'
# model3='RF'
# model4='Adaboost'

# classifiers=[classifier1,classifier2,classifier3,classifier4]
# models=[model1,model2,model3,model4]

# for idx, classifier in enumerate(classifiers):

#     results=my_kfold_cross_validation(kfold_value=10, classifier=classifier, dataset=dataset, labels_binary=labels)
#     plot_roc_curve(labels_test=labels,results_classifier=results,model=models[idx],timewindow=windowtime)


# # #SECTION - S I M U L A T I O N 15

# data="FPS_ICE-kfold10-Binary"
# hash="nilsimsa" 
# windowtime="15"

# dataset_path="/home/mireya/Documentos/Datasets-IoT/ICEdataset/ICEdataset_FPs_binary/dataset_%ss_nilsimsa.pkl"%(windowtime)
# with open(dataset_path,'rb') as f:
#            dataset=pickle.load(f)

# labels_path="/home/mireya/Documentos/Datasets-IoT/ICEdataset/ICEdataset_FPs_binary/labels_%ss_nilsimsa.pkl"%(windowtime)
# with open(labels_path,'rb') as f:
#            labels=pickle.load(f)

# classifier2 = DecisionTreeClassifier(random_state=0,max_depth=34)
# classifier1 = LogisticRegression(random_state=0,max_iter=125)
# classifier3 = RandomForestClassifier(random_state=0,max_depth=18,n_estimators=175)
# classifier4 = AdaBoostClassifier(random_state=0,n_estimators=200)
# model2='DT'
# model1='LR'
# model3='RF'
# model4='Adaboost'

# classifiers=[classifier1,classifier2,classifier3,classifier4]
# models=[model1,model2,model3,model4]

# for idx, classifier in enumerate(classifiers):

#     results=my_kfold_cross_validation(kfold_value=10, classifier=classifier, dataset=dataset, labels_binary=labels)
#     plot_roc_curve(labels_test=labels,results_classifier=results,model=models[idx],timewindow=windowtime)    

# #SECTION - S I M U L A T I O N 20

data="FPS_ICE-kfold10-Binary"
hash="nilsimsa" 
windowtime="20"

dataset_path="/home/mireya/Documentos/Datasets-IoT/ICEdataset/ICEdataset_FPs_binary/dataset_%ss_nilsimsa.pkl"%(windowtime)
with open(dataset_path,'rb') as f:
           dataset=pickle.load(f)

labels_path="/home/mireya/Documentos/Datasets-IoT/ICEdataset/ICEdataset_FPs_binary/labels_%ss_nilsimsa.pkl"%(windowtime)
with open(labels_path,'rb') as f:
           labels=pickle.load(f)

classifier2 = DecisionTreeClassifier(random_state=0,max_depth=36)
classifier1 = LogisticRegression(random_state=0,max_iter=25)
classifier3 = RandomForestClassifier(random_state=0,max_depth=17,n_estimators=287)
classifier4 = AdaBoostClassifier(random_state=0,n_estimators=200)
model2='DT'
model1='LR'
model3='RF'
model4='Adaboost'

classifiers=[classifier1,classifier2,classifier3,classifier4]
models=[model1,model2,model3,model4]

for idx, classifier in enumerate(classifiers):

    results=my_kfold_cross_validation(kfold_value=10, classifier=classifier, dataset=dataset, labels_binary=labels)
    plot_roc_curve(labels_test=labels,results_classifier=results,model=models[idx],timewindow=windowtime)    
