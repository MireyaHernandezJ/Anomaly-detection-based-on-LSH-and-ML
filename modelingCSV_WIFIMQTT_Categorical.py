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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import numpy as np

def categoricalDataset_Train():
   
    #class 0 normal
    trainB=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/Benign_train.pcap.csv")

    labels=[0]*(trainB.shape[0])
    norm_train=trainB.shape[0]

    #class 1 spoofinng=ARP
    train1=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/ARP_Spoofing_train.csv")
    ARP_train=train1.shape[0]
    labels.extend([1]*(ARP_train))


    #class 2 recon
    train2=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/Recon-OS_Scan_train.pcap.csv")
    train3=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/Recon-Ping_Sweep_train.pcap.csv")
    train4=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/Recon-Port_Scan_train.pcap.csv")
    train5=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/Recon-VulScan_train.pcap.csv")
    Recon_train=train2.shape[0]+train3.shape[0]+train4.shape[0]+train5.shape[0]
    labels.extend([2]*(Recon_train))

    #class 3 mqtt
    train6=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/MQTT-DDoS-Connect_Flood_train.pcap.csv")
    train7=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/MQTT-DDoS-Publish_Flood_train.pcap.csv") 
    train8=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/MQTT-DoS-Connect_Flood_train.pcap.csv")
    train9=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/MQTT-DoS-Publish_Flood_train.pcap.csv")
    train10=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/train/MQTT-Malformed_Data_train.pcap.csv")
    mqtt_train=train6.shape[0]+train7.shape[0]+train8.shape[0]+train9.shape[0]+train10.shape[0]
    labels.extend([3]*(mqtt_train))

    #class 4 DDoS
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
    DDoS_train=train11.shape[0]+train12.shape[0]+train13.shape[0]+train14.shape[0]+train15.shape[0]+train16.shape[0]+train17.shape[0]+train18.shape[0]+train19.shape[0]+train20.shape[0]+train21.shape[0]+train22.shape[0]+train23.shape[0]+train24.shape[0]+train25.shape[0]+train26.shape[0]+train27.shape[0]+train28.shape[0]+train29.shape[0]+train30.shape[0]+train31.shape[0]+train32.shape[0]+train33.shape[0]+train34.shape[0]
    labels.extend([4]*(DDoS_train))

    #class 5 DoS
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

    DoS_train=train35.shape[0]+train36.shape[0]+train37.shape[0]+train38.shape[0]+train39.shape[0]+train40.shape[0]+train41.shape[0]+train42.shape[0]+train43.shape[0]+train44.shape[0]+train45.shape[0]+train46.shape[0]+train47.shape[0]+train48.shape[0]+train49.shape[0]+train50.shape[0]
    labels.extend([5]*(DoS_train))
    #JOIN csv files
    dataset=pd.concat([trainB,train1,train2,train3,train4,train5,train6,train7,train8,train9,train10,train11,train12,train13,train14,train15,train16,train17,train18,train19,train20,train21,train22,train23,train24,train25,train26,train27,train28,train29,train30,train31,train32,train33,train34,train35,train36,train37,train38,train39,train40,train41,train42,train43,train44,train45,train46,train47,train48,train49,train50],axis=0)


    return(dataset,labels,norm_train,ARP_train,Recon_train,mqtt_train,DDoS_train,DoS_train)



def categoricalDataset_Test():

    #class 0 normal
    testB=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/Benign_test.pcap.csv")
    norm_test=testB.shape[0]
    labels=[0]*norm_test

    #class 1 spoofinng=ARP
    test1=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/ARP_Spoofing_test.pcap.csv")
    ARP_test=test1.shape[0]
    labels.extend([1]*(ARP_test))

    #class 2 recon
    test2=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/Recon-OS_Scan_test.pcap.csv")
    test3=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/Recon-Ping_Sweep_test.pcap.csv")
    test4=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/Recon-Port_Scan_test.pcap.csv")
    test5=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/Recon-VulScan_test.pcap.csv")
    Recon_test=test2.shape[0]+test3.shape[0]+test4.shape[0]+test5.shape[0]
    labels.extend([2]*(Recon_test))



    #class 3 mqtt
    test6=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/MQTT-DDoS-Connect_Flood_test.pcap.csv")
    test7=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/MQTT-DDoS-Publish_Flood_test.pcap.csv") 
    test8=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/MQTT-DoS-Connect_Flood_test.pcap.csv")
    test9=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/MQTT-DoS-Publish_Flood_test.pcap.csv")
    test10=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/MQTT-Malformed_Data_test.pcap.csv")
    mqtt_test=test6.shape[0]+test7.shape[0]+test8.shape[0]+test9.shape[0]+test10.shape[0]
    labels.extend([3]*(mqtt_test))

    #class 4 DDoS
    test11=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/TCP_IP-DDoS-ICMP1_test.pcap.csv")
    test12=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/TCP_IP-DDoS-ICMP2_test.pcap.csv")
    test13=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/TCP_IP-DDoS-SYN_test.pcap.csv")
    test14=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/TCP_IP-DDoS-TCP_test.pcap.csv")
    test15=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/TCP_IP-DDoS-UDP1_test.pcap.csv")
    test16=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/TCP_IP-DDoS-UDP2_test.pcap.csv")
    DDoS_test=test11.shape[0]+test12.shape[0]+test13.shape[0]+test14.shape[0]+test15.shape[0]+test16.shape[0]
    labels.extend([4]*(DDoS_test))

    #class 5 DoS
    test17=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/TCP_IP-DoS-ICMP_test.pcap.csv")
    test18=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/TCP_IP-DoS-SYN_test.pcap.csv")
    test19=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/TCP_IP-DoS-TCP_test.pcap.csv")
    test20=pd.read_csv("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/WIFI-MQTT/csv/test/TCP_IP-DoS-UDP_test.pcap.csv")
    DoS_test=test17.shape[0]+test18.shape[0]+test19.shape[0]+test20.shape[0]
    labels.extend([5]*(DoS_test))
  
  
    #JOIN csv files
    dataset=pd.concat([testB,test1,test2,test3,test4,test5,test6,test7,test8,test9,test10,test11,test12,test13,test14,test15,test16,test17,test18,test19,test20],axis=0)
    
 

    return(dataset,labels,norm_test,ARP_test,Recon_test,mqtt_test,DDoS_test,DoS_test)



train,ltrain,norm_train,ARP_train,Recon_train,mqtt_train,DDoS_train,DoS_train=categoricalDataset_Train()
test,ltest,norm_test,ARP_test,Recon_test,mqtt_test,DDoS_test,DoS_test=categoricalDataset_Test()


data="CSV_WifiMQTT-TrainTestDEFAULT-categorical"
hash="nilsimsa" #tlsh or nilsimsa
windowtime="5s"#["5s","10s","15s","20s"]


#SECTION - S I M U L A T I O N 1 default

# model1 = "OneVsRest"
# classifier1 = OneVsRestClassifier(SVC())
# hyperparameters1= "default"

# model2 = "LogR:Multi"
# classifier2 = LogisticRegression(multi_class='multinomial',solver='lbfgs')
# hyperparameters2= "default"


# model3 = "LogR:OneVsRest"
# classifier3 = LogisticRegression(multi_class='ovr',solver='lbfgs')
# hyperparameters3= "default"


# model4 = "DT"
# classifier4 = DecisionTreeClassifier()
# hyperparameters4= "default"


# model5 = "RF"
# classifier5 = RandomForestClassifier()
# hyperparameters5= "default"


model6 = "Adaboost"
classifier6 = AdaBoostClassifier()
hyperparameters6= "default"
classifiers=[classifier6]
models=[model6]
hyperparameterss=[hyperparameters6]

# classifiers=[classifier1,classifier2,classifier3,classifier4,classifier5,classifier6]
# models=[model1,model2,model3,model4,model5,model6]
# hyperparameterss=[hyperparameters1,hyperparameters2,hyperparameters3,hyperparameters4,hyperparameters5,hyperparameters6]



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

