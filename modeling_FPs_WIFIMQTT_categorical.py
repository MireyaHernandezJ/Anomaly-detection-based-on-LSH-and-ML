import pickle
import csv
import time
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def count_ceros(arr):
    count=0
    for num in arr:
        if num==0:
               count+=1
    return count 
def count_ones(arr):
    count=0
    for num in arr:
        if num==1:
               count+=1
    return count 
def count_twos(arr):
    count=0
    for num in arr:
        if num==2:
               count+=1
    return count 
def count_threes(arr):
    count=0
    for num in arr:
        if num==3:
               count+=1
    return count 
def count_fours(arr):
    count=0
    for num in arr:
        if num==4:
               count+=1
    return count 
def count_fives(arr):
    count=0
    for num in arr:
        if num==5:
               count+=1
    return count 



#DATASET:

train_path="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/Datasets_5s/train_categorical.pkl"
with open(train_path,'rb') as f:
           train=pickle.load(f)


ltrain_path="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/Datasets_5s/label_train_categorical.pkl"
with open(ltrain_path,'rb') as f:
           ltrain=pickle.load(f)

test_path="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/Datasets_5s/test_categorical.pkl"
with open(test_path,'rb') as f:
           test=pickle.load(f)

ltest_path="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/Datasets_5s/label_test_categorical.pkl"
with open(ltest_path,'rb') as f:
           ltest=pickle.load(f)




data="FPS_WifiMQTT-TrainTestDEFAULT-categorical"
hash="nilsimsa" #tlsh or nilsimsa
windowtime="5s"#["5s","10s","15s","20s"]

norm_train=count_ceros(ltrain)
ARP_train=count_ones(ltrain)
Recon_train=count_twos(ltrain)
mqtt_train=count_threes(ltrain)
DDoS_train=count_fours(ltrain)
DoS_train=count_fives(ltrain)

norm_test=count_ceros(ltest)
ARP_test=count_ones(ltest)
Recon_test=count_twos(ltest)
mqtt_test=count_threes(ltest)
DDoS_test=count_fours(ltest)
DoS_test=count_fives(ltest)

#SECTION - S I M U L A T I O N 1 default

model1 = "OneVsRest"
classifier1 = OneVsRestClassifier(SVC())
hyperparameters1= "default"

model2 = "LogR:Multi"
classifier2 = LogisticRegression(multi_class='multinomial',solver='lbfgs')
hyperparameters2= "default"

model3 = "LogR:OneVsRest"
classifier3 = LogisticRegression(multi_class='ovr',solver='lbfgs')
hyperparameters3= "default"

model4 = "DT"
classifier4 = DecisionTreeClassifier()
hyperparameters4= "default"

model5 = "RF"
classifier5 = RandomForestClassifier()
hyperparameters5= "default"


model6 = "Adaboost"
classifier6 = AdaBoostClassifier()
hyperparameters6= "default"


classifiers=[classifier1,classifier2,classifier3,classifier4,classifier5,classifier6]
models=[model1,model2,model3,model4,model5,model6]
hyperparameterss=[hyperparameters1,hyperparameters2,hyperparameters3,hyperparameters4,hyperparameters5,hyperparameters6]



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

    

    cm=confusion_matrix(ltest,pred)
    cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    
    acc=cm.diagonal()*100
    prec_attack=precision_score(ltest,pred,average=None)*100
    rec_attack=recall_score(ltest,pred,average=None)*100
    f1_attack=f1_score(ltest,pred,average=None)*100
    

    new_row=[
        data,
        norm_train,
        norm_test,
        ARP_train,
        ARP_test,
        Recon_train,
        Recon_test,
        mqtt_train,
        mqtt_test,
        DDoS_train,
        DDoS_test,
        DoS_train,
        DoS_test,
        models[idx],
        hyperparameterss[idx],
        hash,
        windowtime,
        acc,
        prec_attack,
        rec_attack,
        f1_attack,
        training_time,
    ]

    # Open the CSV file in append mode
    with open(
            "/home/mireya/Documentos/Paper_3/Results_CICIoMT2024/results_WIFI-MQTT_categorical.csv",
            mode="a",
            newline="",
    ) as csvfile:
        # Create a CSV writer
        writer = csv.writer(csvfile)
        # Write the new row to the CSV file
        writer.writerow(new_row)

