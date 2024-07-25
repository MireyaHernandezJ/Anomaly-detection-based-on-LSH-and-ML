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

def count_ceros(arr):
    count=0
    for num in arr:
        if num==0:
               count+=1
    return count 

#DATASET:

train_path="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/Datasets_5s/train_binary.pkl"
with open(train_path,'rb') as f:
           train=pickle.load(f)


ltrain_path="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/Datasets_5s/label_train_binary.pkl"
with open(ltrain_path,'rb') as f:
           ltrain=pickle.load(f)


normal_train_samples=count_ceros(ltrain)
attack_train_samples=len(ltrain)-count_ceros(ltrain)


test_path="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/Datasets_5s/test_binary.pkl"
with open(test_path,'rb') as f:
           test=pickle.load(f)

ltest_path="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/Datasets_5s/label_test_binary.pkl"
with open(ltest_path,'rb') as f:
           ltest=pickle.load(f)

normal_test_samples=count_ceros(ltest)
attack_test_samples=len(ltest)-count_ceros(ltest)


data="FPS_WifiMQTT-TrainTestDEFAULT-Binary"
hash="nilsimsa" #tlsh or nilsimsa
windowtime="5s"#["5s","10s","15s","20s"]
samples_train=len(train)
samples_test=len(test)

#### C L A S S I F I E R S ####

### DECISION TREE ###
# model = "DT"
# clf=DecisionTreeClassifier()
# param_grid={'max_depth':[21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]}

### RANDOM FOREST ###
# model = "RF"
# clf=RandomForestClassifier()
# param_grid={'n_estimators':[25,50,100,125,150,175,200,225,250,275,300],'max_depth':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]}

# ### Logistic Regression ###
# model = "LR"
# clf=LogisticRegression()
# param_grid={'max_iter':[25,50,100,125,150,150,175,200,225,250,275,300]}
# param_grid={'max_iter':[155,160,165,170,175,180,185,190,195]}

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
        "/home/mireya/Documentos/Paper_3/Results_CICIoMT2024/results_WIFI-MQTT_GRIDSEARCH_%s.csv"%(model),
        mode="a",
        newline="",
) as csvfile:
    # Create a CSV writer
    writer = csv.writer(csvfile)
    # Write the new row to the CSV file
    writer.writerow(new_row)

