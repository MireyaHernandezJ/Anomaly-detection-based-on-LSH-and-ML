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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split


def count_ceros(arr):
    count=0
    for num in arr:
        if num==0:
               count+=1
    return count 


def ReadDataset():
   
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
    dataset1=pd.concat([train1,train2,train3,train4,train5,train6,train7,train8,train9,train10,train11,train12,train13,train14,train15,train16,train17,train18,test1,test2,test3,test4,test5,test6,test7,test8,test9,test10,test11,test12,test19,test23,test27,test28,test35,test39,test43,test47,train19,train20,train21,train22,train23,train24,train25,train26,train27,train28,train29,train30,train31,train32,train33,train34,train35,train36,train37,train38,train39,train40,train41,train42,train43,train44,train45,train46,train47,train48,train49,train50],axis=0)
    labels=[1]*(dataset1.shape[0])
   

    trainB=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/Benign_train.pcap.csv")
    testB=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/Benign_test.pcap.csv")
    dataset=pd.concat([dataset1,trainB,testB],axis=0)
    labels.extend([0]*((trainB.shape[0])+(testB.shape[0])))
  

    return(dataset,labels)

dataset, labels=ReadDataset()
train,test,ltrain,ltest=train_test_split(dataset,labels,test_size=0.2,random_state=42)



normal_train_samples=count_ceros(ltrain)
attack_train_samples=len(ltrain)-count_ceros(ltrain)

normal_test_samples=count_ceros(ltest)
attack_test_samples=len(ltest)-count_ceros(ltest)


data="CSVs_WifiMQTT-TrainTestNEW-Binary"
hash="None"
windowtime="None"

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

#model1 = "RF"
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

classifiers = [classifier3]
models = [model3]
hyperparameterss= [hyperparameters3]

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
