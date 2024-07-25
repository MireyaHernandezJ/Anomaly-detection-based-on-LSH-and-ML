import pandas as pd
import csv
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

def binaryDataset_Train():
   
    train1=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/ARP_Spoofing_train.csv")
    train2=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/MQTT-DDoS-Connect_Flood_train.pcap.csv")
    train3=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/MQTT-DDoS-Publish_Flood_train.pcap.csv") 
    train4=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/MQTT-DoS-Connect_Flood_train.pcap.csv")
    train5=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/MQTT-DoS-Publish_Flood_train.pcap.csv")
    train6=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/MQTT-Malformed_Data_train.pcap.csv")
    train7=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/Recon-OS_Scan_train.pcap.csv")
    train8=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/Recon-Ping_Sweep_train.pcap.csv")
    train9=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/Recon-Port_Scan_train.pcap.csv")
    train10=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/Recon-VulScan_train.pcap.csv")
    train11=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DDoS-ICMP1_train.pcap.csv")
    train12=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DDoS-ICMP2_train.pcap.csv")
    train13=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DDoS-ICMP3_train.pcap.csv")
    train14=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DDoS-ICMP4_train.pcap.csv")
    train15=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DDoS-ICMP5_train.pcap.csv")
    train16=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DDoS-ICMP6_train.pcap.csv")
    train17=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DDoS-ICMP7_train.pcap.csv")
    train18=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DDoS-ICMP8_train.pcap.csv")
    train19=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DDoS-SYN1_train.pcap.csv")
    train20=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DDoS-SYN2_train.pcap.csv")
    train21=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DDoS-SYN3_train.pcap.csv")
    train22=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DDoS-SYN4_train.pcap.csv")
    train23=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DDoS-TCP1_train.pcap.csv")
    train24=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DDoS-TCP2_train.pcap.csv")
    train25=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DDoS-TCP3_train.pcap.csv")
    train26=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DDoS-TCP4_train.pcap.csv")
    train27=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DDoS-UDP1_train.pcap.csv")
    train28=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DDoS-UDP2_train.pcap.csv")
    train29=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DDoS-UDP3_train.pcap.csv")
    train30=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DDoS-UDP4_train.pcap.csv")
    train31=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DDoS-UDP5_train.pcap.csv")
    train32=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DDoS-UDP6_train.pcap.csv")
    train33=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DDoS-UDP7_train.pcap.csv")
    train34=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DDoS-UDP8_train.pcap.csv")
    train35=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DoS-ICMP1_train.pcap.csv")
    train36=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DoS-ICMP2_train.pcap.csv")
    train37=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DoS-ICMP3_train.pcap.csv")
    train38=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DoS-ICMP4_train.pcap.csv")
    train39=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DoS-SYN1_train.pcap.csv")
    train40=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DoS-SYN2_train.pcap.csv")
    train41=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DoS-SYN3_train.pcap.csv")
    train42=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DoS-SYN4_train.pcap.csv")
    train43=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DoS-TCP1_train.pcap.csv")
    train44=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DoS-TCP2_train.pcap.csv")
    train45=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DoS-TCP3_train.pcap.csv")
    train46=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DoS-TCP4_train.pcap.csv")
    train47=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DoS-UDP1_train.pcap.csv")
    train48=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DoS-UDP2_train.pcap.csv")
    train49=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DoS-UDP3_train.pcap.csv")
    train50=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/TCP_IP-DoS-UDP4_train.pcap.csv")
      
    #JOIN csv files
    dataset1=pd.concat([train1,train2,train3,train4,train5,train6,train7,train8,train9,train10,train11,train12,train13,train14,train15,train16,train17,train18,train19,train20,train21,train22,train23,train24,train25,train26,train27,train28,train29,train30,train31,train32,train33,train34,train35,train36,train37,train38,train39,train40,train41,train42,train43,train44,train45,train46,train47,train48,train49,train50],axis=0)
    labels=[1]*(dataset1.shape[0])
    n_label_attack=len(labels)

    trainB=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/Benign_train.pcap.csv")
    dataset=pd.concat([dataset1,trainB],axis=0)
    labels.extend([0]*(trainB.shape[0]))
    n_label_normal=trainB.shape[0]

    return(dataset,labels,n_label_normal,n_label_attack)



def binaryDataset_Test():
   
    test1=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/ARP_Spoofing_test.pcap.csv")
    test2=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/MQTT-DDoS-Connect_Flood_test.pcap.csv")
    test3=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/MQTT-DDoS-Publish_Flood_test.pcap.csv") 
    test4=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/MQTT-DoS-Connect_Flood_test.pcap.csv")
    test5=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/MQTT-DoS-Publish_Flood_test.pcap.csv")
    test6=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/MQTT-Malformed_Data_test.pcap.csv")
    test7=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/Recon-OS_Scan_test.pcap.csv")
    test8=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/Recon-Ping_Sweep_test.pcap.csv")
    test9=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/Recon-Port_Scan_test.pcap.csv")
    test10=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/Recon-VulScan_test.pcap.csv")
    test11=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/TCP_IP-DDoS-ICMP1_test.pcap.csv")
    test12=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/TCP_IP-DDoS-ICMP2_test.pcap.csv")
    test19=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/TCP_IP-DDoS-SYN_test.pcap.csv")
    test23=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/TCP_IP-DDoS-TCP_test.pcap.csv")
    test27=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/TCP_IP-DDoS-UDP1_test.pcap.csv")
    test28=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/TCP_IP-DDoS-UDP2_test.pcap.csv")
    test35=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/TCP_IP-DoS-ICMP_test.pcap.csv")
    test39=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/TCP_IP-DoS-SYN_test.pcap.csv")
    test43=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/TCP_IP-DoS-TCP_test.pcap.csv")
    test47=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/TCP_IP-DoS-UDP_test.pcap.csv")
  
  
    #JOIN csv files
    dataset1=pd.concat([test1,test2,test3,test4,test5,test6,test7,test8,test9,test10,test11,test12,test19,test23,test27,test28,test35,test39,test43,test47],axis=0)
    labels=[1]*(dataset1.shape[0])
    n_label_attack=len(labels)

    testB=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/Benign_test.pcap.csv")
    dataset=pd.concat([dataset1,testB],axis=0)
    labels.extend([0]*(testB.shape[0]))
    n_label_normal=testB.shape[0]

    return(dataset,labels,n_label_normal,n_label_attack)

train,ltrain,normal_train_samples,attack_train_samples=binaryDataset_Train()
test,ltest,normal_test_samples,attack_test_samples=binaryDataset_Test()

samples_train=train.shape[0]
samples_test=test.shape[0]

data="CSVs_WifiMQTT-TrainTestDEFAULT-Binary"
hash="None"
windowtime="None"


#USING GRID SEARCH
model = "DT"
clf=DecisionTreeClassifier()
#param_grid={'max_depth':[1,2,3,4,5]}
param_grid={'max_depth':[6,7,8,9,10]}

#### ACCURACY ####
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy')
grid_search.fit(train, ltrain)

# Get the best parameters
best_params = grid_search.best_params_
hyperparameters_acc = "Best max Depth:%s"%(best_params)

# Use the best model to make predictions
best_clf = grid_search.best_estimator_
results = best_clf.predict(test)
accuracy = accuracy_score(ltest, results)*100


#### PRECISION ####
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='precision')
grid_search.fit(train, ltrain)

# Get the best parameters
best_params = grid_search.best_params_
hyperparameters_prec = "Best max Depth:%s"%(best_params)

# Use the best model to make predictions
best_clf = grid_search.best_estimator_
results = best_clf.predict(test)
precision = precision_score(ltest,results)*100

#### RECALL ####
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='recall')
grid_search.fit(train, ltrain)

# Get the best parameters
best_params = grid_search.best_params_
hyperparameters_rec = "Best max Depth:%s"%(best_params)

# Use the best model to make predictions
best_clf = grid_search.best_estimator_
results = best_clf.predict(test)
recall = recall_score(ltest,results)*100

#### F1-SCORE ####
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='f1')
grid_search.fit(train, ltrain)

# Get the best parameters
best_params = grid_search.best_params_
hyperparameters_f1 = "Best max Depth:%s"%(best_params)

# Use the best model to make predictions
best_clf = grid_search.best_estimator_
results = best_clf.predict(test)
f1 = f1_score(ltest,results)*100


#### AUC ####
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='roc_auc')
grid_search.fit(train, ltrain)

# Get the best parameters
best_params = grid_search.best_params_
hyperparameters_auc = "Best max Depth:%s"%(best_params)

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

