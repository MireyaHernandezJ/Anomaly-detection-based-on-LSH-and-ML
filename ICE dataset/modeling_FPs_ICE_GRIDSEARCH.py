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
from sklearn.model_selection import GridSearchCV
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
hash="nilsimsa" #tlsh or nilsimsa
windowtime="20"#["5s","10s","15s","20s"]
dataset_path="/home/mireya/Documentos/Datasets-IoT/ICEdataset/ICEdataset_FPs_binary/dataset_%ss_nilsimsa.pkl"%(windowtime)
with open(dataset_path,'rb') as f:
           dataset=pickle.load(f)

labels_path="/home/mireya/Documentos/Datasets-IoT/ICEdataset/ICEdataset_FPs_binary/labels_%ss_nilsimsa.pkl"%(windowtime)
with open(labels_path,'rb') as f:
           labels=pickle.load(f)

train,test,ltrain,ltest=train_test_split(dataset,labels,test_size=0.2,random_state=42)

normal_test_samples=count_ceros(ltest)
attack_test_samples=len(ltest)-count_ceros(ltest)
normal_train_samples=count_ceros(ltrain)
attack_train_samples=len(ltrain)-count_ceros(ltrain)
samples_train=len(train)
samples_test=len(test)

#### C L A S S I F I E R S ####

### DECISION TREE ###
# model = "DT"
# clf=DecisionTreeClassifier()
# param_grid={'max_depth':[4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]}

### RANDOM FOREST ###
# model = "RF"
# clf=RandomForestClassifier()
# param_grid={'n_estimators':[5,10,15,20,25,50,100,125,150,175,200,225,250,275,300],'max_depth':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]}

# ### Logistic Regression ###
# model = "LR"
# clf=LogisticRegression()
# param_grid={'max_iter':[25,50,100,125,150,155,160,165,170,175,180,185,190,195,150,175,200,225,250,275,300]}


# ### Adaboost ###
model = "Adaboost"
clf=AdaBoostClassifier()
param_grid={'n_estimators':[10,15,20,25,50,100,125,150,175,200,225,250,275,300]}

########## P E R F O R M A N C E 

#### ACCURACY ####
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy')
grid_search.fit(train, ltrain)

# Get the best parameters
best_params = grid_search.best_params_
hyperparameters_acc = "Best hyper:%s"%(best_params)

# Use the best model to make predictions
best_clf = grid_search.best_estimator_
results = best_clf.predict(test)
accuracy = accuracy_score(ltest, results)*100


#### PRECISION ####
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='precision')
grid_search.fit(train, ltrain)

# Get the best parameters
best_params = grid_search.best_params_
hyperparameters_prec ="Best hyper:%s"%(best_params)

# Use the best model to make predictions
best_clf = grid_search.best_estimator_
results = best_clf.predict(test)
precision = precision_score(ltest,results)*100

#### RECALL ####
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='recall')
grid_search.fit(train, ltrain)

# Get the best parameters
best_params = grid_search.best_params_
hyperparameters_rec = "Best hyper:%s"%(best_params)

# Use the best model to make predictions
best_clf = grid_search.best_estimator_
results = best_clf.predict(test)
recall = recall_score(ltest,results)*100

#### F1-SCORE ####
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='f1')
grid_search.fit(train, ltrain)

# Get the best parameters
best_params = grid_search.best_params_
hyperparameters_f1 = "Best hyper:%s"%(best_params)

# Use the best model to make predictions
best_clf = grid_search.best_estimator_
results = best_clf.predict(test)
f1 = f1_score(ltest,results)*100


#### AUC ####
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='roc_auc')
grid_search.fit(train, ltrain)

# Get the best parameters
best_params = grid_search.best_params_
hyperparameters_auc = "Best hyper:%s"%(best_params)

# Use the best model to make predictions
best_clf = grid_search.best_estimator_
results = best_clf.predict(test)
roc_auc = roc_auc_score(ltest,results)*100



# S A V I N G

new_row=[
    data,
    attack_train_samples,
    normal_train_samples,
    attack_test_samples,
    normal_test_samples,
    samples_train,
    samples_test,
    model,
    hash,
    windowtime,
    accuracy,
    hyperparameters_acc,
    precision,
    hyperparameters_prec,
    recall,
    hyperparameters_rec,
    f1,
    hyperparameters_f1,
    roc_auc,
    hyperparameters_auc,
   
]

# Open the CSV file in append mode
with open(
        "/home/mireya/Documentos/Paper_3/Results_ICEdataset/results_GRIDSEARCH_%s.csv"%(model),
        mode="a",
        newline="",
) as csvfile:
    # Create a CSV writer
    writer = csv.writer(csvfile)
    # Write the new row to the CSV file
    writer.writerow(new_row)

