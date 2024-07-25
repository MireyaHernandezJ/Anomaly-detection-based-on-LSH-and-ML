import pickle
import os
#Esta funcion sirve cuando el pickle es una lista de listas
def join_piclkes(filepaths,savepath):
    joined_data=[]
    for filepath in filepaths:
        with open(filepath,'rb') as f:
            data=pickle.load(f)
            joined_data.extend(data)

    with open(savepath,'wb') as f:
        pickle.dump(joined_data, f)

    return print("Joined")

# path="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/normal/train/20s/nilsimsa_train.pkl"
# path1 = "/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/normal/train/20s/nilsimsa_train1.pkl"
# path2 = "/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/normal/train/20s/nilsimsa_train2.pkl"
# path3 = "/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/normal/train/20s/nilsimsa_train3.pkl"
# filepaths=[path,path1,path2,path3]
# savepath="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/normal/train/20s/nilsimsa.pkl"
# join_piclkes(filepaths,savepath)

def joining(scenario,filename,nsplits,subset,nfiles,subsetsave):
    pickles=[]
    for i in range(1,nfiles+1):
        filepath="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/%s/%s/%s/%ss/%s/nilsimsa_%s_%d.pkl"%(filename,scenario,subsetsave,nsplits,subset,subset,i)
        savepath="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/%s/%s/%s/%ss/nilsimsa_%s_ALL.pkl"%(filename,scenario,subsetsave,nsplits,subset)
        with open(filepath,'rb') as f:
            data=pickle.load(f)
        pickles.extend(data) #probando extend en lugar de append
    with open(savepath,'wb') as f:
        pickle.dump(pickles, f)
    return print("joined")

def joining2(scenario,filename,nsplits,subset,nfiles,subsetsave): #Porque hay PKL files with 0 bytes debido a que no habia paquetes quejalar en ataques DDoS
    pickles=[]
    for i in range(1,nfiles+1):
        filepath="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/%s/%s/%s/%ss/%s/nilsimsa_%s_%d.pkl"%(filename,scenario,subsetsave,nsplits,subset,subset,i)
        savepath="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/%s/%s/%s/%ss/nilsimsa_%s_ALL.pkl"%(filename,scenario,subsetsave,nsplits,subset)
        if not os.path.exists(filepath):
            print(f"File {filepath} does not exist. Skipping...")
            continue
        try:
            with open(filepath,'rb') as f:
                data=pickle.load(f)
            pickles.extend(data)
        
        except EOFError:
            print(f"EOFError: Ran out of input in file {filepath}. Skipping...")
            continue
    
    with open(savepath,'wb') as f:
        pickle.dump(pickles, f)
    return print("joined")


def checking(scenario,filename,nsplits,subset,nfiles,subsetsave):#checando la informacion 
    pickles=[]
    for i in range(1700,1702):
        filepath="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/%s/%s/%s/%ss/%s/nilsimsa_%s_%d.pkl"%(filename,scenario,subsetsave,nsplits,subset,subset,i)
        savepath="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/%s/%s/%s/%ss/nilsimsa_%s_ALL.pkl"%(filename,scenario,subsetsave,nsplits,subset)
        with open(filepath,'rb') as f:
            data=pickle.load(f)
        print(data)

def joiningnormal(scenario,filename,nsplits,nfiles,subsetsave): #para el escenario normal del dataset WIFI-MQTT
    pickles=[]
    for i in range(1,n_files+1):
        filepath="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/%s/%s/%s/%ss/nilsimsa_%d.pkl"%(filename,scenario,subsetsave,nsplits,i)
        savepath="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/%s/%s/%s/%ss/nilsimsa_%s_ALL.pkl"%(filename,scenario,subsetsave,nsplits,scenario)
        with open(filepath,'rb') as f:
            data=pickle.load(f)
        pickles.extend(data)
    with open(savepath,'wb') as f:
        pickle.dump(pickles, f)
    return print("joined")

# n_splits="5" #5,10,15,20
# filename="splits_Wifi-MQTT" #splits_Bluetooth or splits_Wifi-MQTT
# scenario="normal" #normal or attack 
# subsetsave="train" # test or train , where to save the FPs
# subset="TCP-IP-DoS-UDP4" #this can be ARP, MQTT-DoS, etc
# directory = "/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/%s/%s/%s/%ss/%s"%(filename,scenario,subsetsave,n_splits,subset)
# directory = "/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/%s/%s/%s/%ss"%(filename,scenario,subsetsave,n_splits) #NORMAL SCENARIO
# n_files = len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
#joining(scenario=scenario,filename=filename,nsplits=n_splits,subset=subset,nfiles=n_files,subsetsave=subsetsave)
#joining2(scenario=scenario,filename=filename,nsplits=n_splits,subset=subset,nfiles=n_files,subsetsave=subsetsave)
#joiningnormal(scenario,filename,n_splits,n_files,subsetsave)


def binaryDataset_TEST():
    test=[]
    Attack_test1="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_ARP_ALL.pkl"
    with open(Attack_test1,'rb') as f:
            data=pickle.load(f)
    test.extend(data)
    Attack_test2="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_MQTT-DDoS-Connect_ALL.pkl"
    with open(Attack_test2,'rb') as f:
            data=pickle.load(f)
    test.extend(data)
    Attack_test3="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_MQTT-DDoS-Publish_ALL.pkl"
    with open(Attack_test3,'rb') as f:
            data=pickle.load(f)
    test.extend(data)
    Attack_test4="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_MQTT-DoS-connect_ALL.pkl"
    with open(Attack_test4,'rb') as f:
            data=pickle.load(f)
    test.extend(data)
    Attack_test5="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_MQTT-DoS-Publish_ALL.pkl"
    with open(Attack_test5,'rb') as f:
            data=pickle.load(f)
    test.extend(data)
    Attack_test6="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_MQTT-Malformed_ALL.pkl"
    with open(Attack_test6,'rb') as f:
            data=pickle.load(f)
    test.extend(data)
    Attack_test7="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_Recon_ALL.pkl"
    with open(Attack_test7,'rb') as f:
            data=pickle.load(f)
    test.extend(data)
    Attack_test8="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-ICMP1_ALL.pkl"
    with open(Attack_test8,'rb') as f:
            data=pickle.load(f)
    test.extend(data)
    Attack_test9="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-ICMP2_ALL.pkl"
    with open(Attack_test9,'rb') as f:
            data=pickle.load(f)
    test.extend(data)
    Attack_test10="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-SYN_ALL.pkl"
    with open(Attack_test10,'rb') as f:
            data=pickle.load(f)
    test.extend(data)
    Attack_test11="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-TCP_ALL.pkl"
    with open(Attack_test11,'rb') as f:
            data=pickle.load(f)
    test.extend(data)
    Attack_test12="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-UDP1_ALL.pkl"
    with open(Attack_test12,'rb') as f:
            data=pickle.load(f)
    test.extend(data)
    Attack_test13="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-UDP2_ALL.pkl"
    with open(Attack_test13,'rb') as f:
            data=pickle.load(f)
    test.extend(data)
    Attack_test14="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-ICMP_ALL.pkl"
    with open(Attack_test14,'rb') as f:
            data=pickle.load(f)
    test.extend(data)
    Attack_test15="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-SYN_ALL.pkl"
    with open(Attack_test15,'rb') as f:
            data=pickle.load(f)
    test.extend(data)
    Attack_test16="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-TCP_ALL.pkl"
    with open(Attack_test16,'rb') as f:
            data=pickle.load(f)
    test.extend(data)
    Attack_test17="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-UDP_ALL.pkl"
    with open(Attack_test17,'rb') as f:
            data=pickle.load(f)
    test.extend(data)

    labels=[1]*len(test)

    Normal_test="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/normal/test/5s/nilsimsa_normal_ALL.pkl"
    with open(Normal_test,'rb') as f:
            data=pickle.load(f)
    test.extend(data)
        
    label_normal=[0]*(len(test)-len(labels))    
    labels.extend(label_normal)

    savepath="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/test_binary.pkl"
    with open(savepath,'wb') as f:
        pickle.dump(test, f)
    savepath="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/label_test_binary.pkl"
    with open(savepath,'wb') as f:
        pickle.dump(labels, f)
    return print("saved")



def binaryDataset_Train():
    train=[]
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_ARP_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_MQTT-DDoS-Connect-Flood_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_MQTT-DDoS-Publish-Flood_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_MQTT-DoS-Connect-Flood_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_MQTT-DoS-Publish-Flood_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_MQTT-malformed_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_Recon_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-ICMP1_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-ICMP2_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-ICMP3_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-ICMP4_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-ICMP5_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-ICMP6_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-ICMP7_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-ICMP8_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-SYN1_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-SYN2_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-SYN3_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-SYN4_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-TCP1_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-TCP2_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-TCP3_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-TCP4_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)


    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-UDP1_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-UDP2_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-UDP3_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-UDP4_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-UDP5_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-UDP6_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-UDP7_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-UDP8_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)



    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-ICMP1_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-ICMP2_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-ICMP3_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-ICMP4_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)


    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-SYN1_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-SYN2_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-SYN3_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-SYN4_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)


    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-TCP1_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-TCP2_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-TCP3_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-TCP4_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)


    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-UDP1_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-UDP2_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-UDP3_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-UDP4_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    labels=[1]*len(train)

    Normal_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/normal/train/5s/nilsimsa_normal_ALL.pkl"
    with open(Normal_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
        
    label_normal=[0]*(len(train)-len(labels))    
    labels.extend(label_normal)

    savepath="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/train_binary.pkl"
    with open(savepath,'wb') as f:
        pickle.dump(train, f)
    savepath="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/label_train_binary.pkl"
    with open(savepath,'wb') as f:
        pickle.dump(labels, f)
    return print("saved")

###################### MULTICLASS : 19 CLASSES
def MultiClassDataset_TEST():
    test=[]
    Attack_test1="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_ARP_ALL.pkl"
    with open(Attack_test1,'rb') as f:
            data=pickle.load(f)
    test.extend(data)
    
    labels=[1]*len(data)

    Attack_test2="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_MQTT-DDoS-Connect_ALL.pkl"
    with open(Attack_test2,'rb') as f:
            data=pickle.load(f)
    test.extend(data)

    labels.extend([2]*len(data))


    Attack_test3="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_MQTT-DDoS-Publish_ALL.pkl"
    with open(Attack_test3,'rb') as f:
            data=pickle.load(f)
    test.extend(data)

    labels.extend([3]*len(data))


    Attack_test4="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_MQTT-DoS-connect_ALL.pkl"
    with open(Attack_test4,'rb') as f:
            data=pickle.load(f)
    test.extend(data)

    labels.extend([4]*len(data))

    Attack_test5="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_MQTT-DoS-Publish_ALL.pkl"
    with open(Attack_test5,'rb') as f:
            data=pickle.load(f)
    test.extend(data)

    labels.extend([5]*len(data))


    Attack_test6="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_MQTT-Malformed_ALL.pkl"
    with open(Attack_test6,'rb') as f:
            data=pickle.load(f)
    test.extend(data)

    labels.extend([6]*len(data))


    Attack_test7="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_Recon_ALL.pkl"
    with open(Attack_test7,'rb') as f:
            data=pickle.load(f)
    test.extend(data)

    labels.extend([7]*len(data))


    Attack_test8="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-ICMP1_ALL.pkl"
    with open(Attack_test8,'rb') as f:
            data=pickle.load(f)
    test.extend(data)

    lenICMP1=len(data)


    Attack_test9="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-ICMP2_ALL.pkl"
    with open(Attack_test9,'rb') as f:
            data=pickle.load(f)
    test.extend(data)

    lenICMP2=len(data)
    labels.extend([8]*(lenICMP1+lenICMP2))

    Attack_test10="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-SYN_ALL.pkl"
    with open(Attack_test10,'rb') as f:
            data=pickle.load(f)
    test.extend(data)

    labels.extend([9]*len(data))


    Attack_test11="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-TCP_ALL.pkl"
    with open(Attack_test11,'rb') as f:
            data=pickle.load(f)
    test.extend(data)

    labels.extend([10]*len(data))


    Attack_test12="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-UDP1_ALL.pkl"
    with open(Attack_test12,'rb') as f:
            data=pickle.load(f)
    test.extend(data)

    lenUDP1=len(data)


    Attack_test13="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-UDP2_ALL.pkl"
    with open(Attack_test13,'rb') as f:
            data=pickle.load(f)
    test.extend(data)

    lenUDP2=len(data)

    labels.extend([11]*(lenUDP1+lenUDP2))


    Attack_test14="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-ICMP_ALL.pkl"
    with open(Attack_test14,'rb') as f:
            data=pickle.load(f)
    test.extend(data)

    labels.extend([12]*len(data))


    Attack_test15="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-SYN_ALL.pkl"
    with open(Attack_test15,'rb') as f:
            data=pickle.load(f)
    test.extend(data)

    labels.extend([13]*len(data))


    Attack_test16="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-TCP_ALL.pkl"
    with open(Attack_test16,'rb') as f:
            data=pickle.load(f)
    test.extend(data)

    labels.extend([14]*len(data))


    Attack_test17="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-UDP_ALL.pkl"
    with open(Attack_test17,'rb') as f:
            data=pickle.load(f)
    test.extend(data)

    labels.extend([15]*len(data))

    Normal_test="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/normal/test/5s/nilsimsa_normal_ALL.pkl"
    with open(Normal_test,'rb') as f:
            data=pickle.load(f)
    test.extend(data)
        
    labels.extend([0]*len(data))

    savepath="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/test_multiclass.pkl"
    with open(savepath,'wb') as f:
        pickle.dump(test, f)
    savepath="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/label_test_multiclass.pkl"
    with open(savepath,'wb') as f:
        pickle.dump(labels, f)
    return print("saved")


def MulticlassDataset_Train():
    train=[]
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_ARP_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    labels=([1]*len(data))

    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_MQTT-DDoS-Connect-Flood_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    labels.extend([2]*len(data))

    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_MQTT-DDoS-Publish-Flood_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    labels.extend([3]*len(data))


    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_MQTT-DoS-Connect-Flood_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    labels.extend([4]*len(data))

    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_MQTT-DoS-Publish-Flood_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    labels.extend([5]*len(data))


    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_MQTT-malformed_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    labels.extend([6]*len(data))


    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_Recon_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    labels.extend([7]*len(data))


    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-ICMP1_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lenICMP1=len(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-ICMP2_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lenICMP2=len(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-ICMP3_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lenICMP3=len(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-ICMP4_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lenICMP4=len(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-ICMP5_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lenICMP5=len(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-ICMP6_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lenICMP6=len(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-ICMP7_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lenICMP7=len(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-ICMP8_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lenICMP8=len(data)
    labels.extend([8]*(lenICMP1+lenICMP2+lenICMP3+lenICMP4+lenICMP5+lenICMP6+lenICMP7+lenICMP8))

    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-SYN1_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lenSYN1=len(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-SYN2_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lenSYN2=len(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-SYN3_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lenSYN3=len(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-SYN4_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lenSYN4=len(data)

    labels.extend([9]*(lenSYN1+lenSYN2+lenSYN3+lenSYN4))
    
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-TCP1_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lenTCP1=len(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-TCP2_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lenTCP2=len(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-TCP3_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lenTCP3=len(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-TCP4_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lenTCP4=len(data)

    labels.extend([10]*( lenTCP1+ lenTCP2+ lenTCP3+ lenTCP4))

    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-UDP1_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lenUDP1=len(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-UDP2_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lenUDP2=len(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-UDP3_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lenUDP3=len(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-UDP4_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lenUDP4=len(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-UDP5_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lenUDP5=len(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-UDP6_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lenUDP6=len(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-UDP7_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lenUDP7=len(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-UDP8_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lenUDP8=len(data)

    labels.extend([11]*(lenUDP1+lenUDP2+lenUDP3+lenUDP4+lenUDP5+lenUDP6+lenUDP7+lenUDP8))


    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-ICMP1_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lenicmp1=len(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-ICMP2_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lenicmp2=len(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-ICMP3_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lenicmp3=len(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-ICMP4_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lenicmp4=len(data)
    labels.extend([12]*(lenicmp1+lenicmp2+lenicmp3+lenicmp4))


    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-SYN1_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lensyn1=len(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-SYN2_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lensyn2=len(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-SYN3_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lensyn3=len(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-SYN4_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lensyn4=len(data)

    labels.extend([13]*(lensyn1+lensyn2+lensyn3+lensyn4))

    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-TCP1_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lentcp1=len(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-TCP2_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lentcp2=len(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-TCP3_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lentcp3=len(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-TCP4_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lentcp4=len(data)

    labels.extend([14]*(lentcp1+lentcp2+lentcp3+lentcp4))

    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-UDP1_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lenudp1=len(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-UDP2_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lenudp2=len(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-UDP3_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lenudp3=len(data)
    Attack_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-UDP4_ALL.pkl"
    with open(Attack_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    lenudp4=len(data)

    labels.extend([15]*(lenudp1+lenudp2+lenudp3+lenudp4))


    Normal_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/normal/train/5s/nilsimsa_normal_ALL.pkl"
    with open(Normal_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
        
    labels.extend([0]*len(data))

    savepath="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/train_multiclass.pkl"
    with open(savepath,'wb') as f:
        pickle.dump(train, f)
    savepath="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/label_train_multiclass.pkl"
    with open(savepath,'wb') as f:
        pickle.dump(labels, f)
    return print("saved")

def joindatasets():
#     dataset_binary=[]
#     labels_binary=[]

#     path_test_binary="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/test_binary.pkl"
#     with open(path_test_binary,'rb') as f:
#            data=pickle.load(f)
#     dataset_binary.extend(data)

#     path_train_binary="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/train_binary.pkl"
#     with open(path_train_binary,'rb') as f:
#            data=pickle.load(f)
#     dataset_binary.extend(data)

#     path_label_test_binary="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/label_test_binary.pkl"
#     with open(path_label_test_binary,'rb') as f:
#            data=pickle.load(f)
#     labels_binary.extend(data)

#     path_label_train_binary="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/label_train_binary.pkl"
#     with open(path_label_train_binary,'rb') as f:
#            data=pickle.load(f)
#     labels_binary.extend(data)



    dataset_multiclass=[]
    labels_multiclass=[]

    path_test_multiclass="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/test_multiclass.pkl"
    with open(path_test_multiclass,'rb') as f:
           data=pickle.load(f)
    dataset_multiclass.extend(data)

    path_train_multiclass="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/train_multiclass.pkl"
    with open(path_train_multiclass,'rb') as f:
           data=pickle.load(f)
    dataset_multiclass.extend(data)

    path_label_test_multiclass="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/label_test_multiclass.pkl"
    with open(path_label_test_multiclass,'rb') as f:
           data=pickle.load(f)
    labels_multiclass.extend(data)

    path_label_train_multiclass="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/label_train_multiclass.pkl"
    with open(path_label_train_multiclass,'rb') as f:
           data=pickle.load(f)
    labels_multiclass.extend(data)


    savepath="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/dataset_binary.pkl"
    with open(savepath,'wb') as f:
        pickle.dump(dataset_binary, f)
    savepath="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/labels_binary.pkl"
    with open(savepath,'wb') as f:
        pickle.dump(labels_binary, f)
    savepath="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/dataset_multiclass.pkl"
    with open(savepath,'wb') as f:
        pickle.dump(dataset_multiclass, f)
    savepath="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/labels_multiclass.pkl"
    with open(savepath,'wb') as f:
        pickle.dump(labels_multiclass, f)
    return print("saved")







########################## CATEGORICAL

def categoricalDataset_TEST():
    test=[]
    Attack_test1="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_ARP_ALL.pkl"
    with open(Attack_test1,'rb') as f:
            data=pickle.load(f)
    test.extend(data)
    
    labels=[1]*len(data) # CLASS SPOOFING 1

    Attack_test7="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_Recon_ALL.pkl"
    with open(Attack_test7,'rb') as f:
            data=pickle.load(f)
    test.extend(data)

    labels.extend([2]*len(data)) #CLASS RECON 2


    Attack_test2="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_MQTT-DDoS-Connect_ALL.pkl"
    with open(Attack_test2,'rb') as f:
            data=pickle.load(f)
    test.extend(data)

    lenMQTT1=len(data)

    Attack_test3="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_MQTT-DDoS-Publish_ALL.pkl"
    with open(Attack_test3,'rb') as f:
            data=pickle.load(f)
    test.extend(data)

    lenMQTT2=len(data)


    Attack_test4="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_MQTT-DoS-connect_ALL.pkl"
    with open(Attack_test4,'rb') as f:
            data=pickle.load(f)
    test.extend(data)

    lenMQTT3=len(data)

    Attack_test5="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_MQTT-DoS-Publish_ALL.pkl"
    with open(Attack_test5,'rb') as f:
            data=pickle.load(f)
    test.extend(data)

    lenMQTT4=len(data)


    Attack_test6="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_MQTT-Malformed_ALL.pkl"
    with open(Attack_test6,'rb') as f:
            data=pickle.load(f)
    test.extend(data)

    lenMQTT5=len(data)
    labels.extend([3]*(lenMQTT1+lenMQTT2+lenMQTT3+lenMQTT4+lenMQTT5)) # CLASS MQTT3 
    
    
    Attack_test8="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-ICMP1_ALL.pkl"
    with open(Attack_test8,'rb') as f:
            data=pickle.load(f)
    test.extend(data)

    lenDDoS1=len(data)


    Attack_test9="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-ICMP2_ALL.pkl"
    with open(Attack_test9,'rb') as f:
            data=pickle.load(f)
    test.extend(data)

    lenDDoS2=len(data)


    Attack_test10="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-SYN_ALL.pkl"
    with open(Attack_test10,'rb') as f:
            data=pickle.load(f)
    test.extend(data)

    lenDDoS3=len(data)


    Attack_test11="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-TCP_ALL.pkl"
    with open(Attack_test11,'rb') as f:
            data=pickle.load(f)
    test.extend(data)

    lenDDoS4=len(data)



    Attack_test12="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-UDP1_ALL.pkl"
    with open(Attack_test12,'rb') as f:
            data=pickle.load(f)
    test.extend(data)

    lenDDoS5=len(data)


    Attack_test13="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-UDP2_ALL.pkl"
    with open(Attack_test13,'rb') as f:
            data=pickle.load(f)
    test.extend(data)

    lenDDoS6=len(data)
    labels.extend([4]*(lenDDoS1+lenDDoS2+lenDDoS3+lenDDoS4+lenDDoS5+lenDDoS6)) #  CLASS DDoS 4


    Attack_test14="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-ICMP_ALL.pkl"
    with open(Attack_test14,'rb') as f:
            data=pickle.load(f)
    test.extend(data)

    lenDoS1=len(data)


    Attack_test15="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-SYN_ALL.pkl"
    with open(Attack_test15,'rb') as f:
            data=pickle.load(f)
    test.extend(data)

    lenDoS2=len(data)


    Attack_test16="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-TCP_ALL.pkl"
    with open(Attack_test16,'rb') as f:
            data=pickle.load(f)
    test.extend(data)

    lenDoS3=len(data)



    Attack_test17="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/test/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-UDP_ALL.pkl"
    with open(Attack_test17,'rb') as f:
            data=pickle.load(f)
    test.extend(data)

    lenDoS4=len(data)
    labels.extend([5]*(lenDoS1+lenDoS2+lenDoS3+lenDoS4)) # CLASS SPOOFING 1 CLASS RECON 2 CLASS MQTT 3 CLASS DDoS 4 CLASS DoS 5


    Normal_test="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/normal/test/5s/nilsimsa_normal_ALL.pkl"
    with open(Normal_test,'rb') as f:
            data=pickle.load(f)
    test.extend(data)
        
    labels.extend([0]*len(data))

    savepath="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/test_categorical.pkl"
    with open(savepath,'wb') as f:
        pickle.dump(test, f)
    savepath="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/label_test_categorical.pkl"
    with open(savepath,'wb') as f:
        pickle.dump(labels, f)
    return print("saved")

def categoricalDataset_Train():
    train=[]
    Attack_train1="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_ARP_ALL.pkl"
    with open(Attack_train1,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
    
    labels=[1]*len(data) # CLASS SPOOFING

    Attack_train7="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_Recon_ALL.pkl"
    with open(Attack_train7,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    labels.extend([2]*len(data)) # CLASS RECON


    Attack_train2="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_MQTT-DDoS-Connect-Flood_ALL.pkl"
    with open(Attack_train2,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenMQTT1=len(data)

    Attack_train3="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_MQTT-DDoS-Publish-Flood_ALL.pkl"
    with open(Attack_train3,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenMQTT2=len(data)


    Attack_train4="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_MQTT-DoS-Connect-Flood_ALL.pkl"
    with open(Attack_train4,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenMQTT3=len(data)

    Attack_train5="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_MQTT-DoS-Publish-Flood_ALL.pkl"
    with open(Attack_train5,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenMQTT4=len(data)


    Attack_train6="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_MQTT-malformed_ALL.pkl"
    with open(Attack_train6,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenMQTT5=len(data)
    labels.extend([3]*(lenMQTT1+lenMQTT2+lenMQTT3+lenMQTT4+lenMQTT5)) # CLASS MQTT
    
    
    Attack_train8="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-ICMP1_ALL.pkl"
    with open(Attack_train8,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDDoS1=len(data)


    Attack_train9="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-ICMP2_ALL.pkl"
    with open(Attack_train9,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDDoS2=len(data)
    
    Attack_train9="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-ICMP3_ALL.pkl"
    with open(Attack_train9,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDDoS3=len(data)

    Attack_train9="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-ICMP4_ALL.pkl"
    with open(Attack_train9,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDDoS4=len(data)

    Attack_train9="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-ICMP5_ALL.pkl"
    with open(Attack_train9,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDDoS5=len(data)

    Attack_train9="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-ICMP6_ALL.pkl"
    with open(Attack_train9,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDDoS6=len(data)

    Attack_train9="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-ICMP7_ALL.pkl"
    with open(Attack_train9,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDDoS7=len(data)

    Attack_train9="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-ICMP8_ALL.pkl"
    with open(Attack_train9,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDDoS8=len(data)


    Attack_train10="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-SYN1_ALL.pkl"
    with open(Attack_train10,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDDoS9=len(data)

    Attack_train10="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-SYN2_ALL.pkl"
    with open(Attack_train10,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDDoS10=len(data)

    Attack_train10="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-SYN3_ALL.pkl"
    with open(Attack_train10,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDDoS11=len(data)

    Attack_train10="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-SYN4_ALL.pkl"
    with open(Attack_train10,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDDoS12=len(data)


    Attack_train11="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-TCP1_ALL.pkl"
    with open(Attack_train11,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDDoS13=len(data)

    Attack_train11="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-TCP2_ALL.pkl"
    with open(Attack_train11,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDDoS14=len(data)

    Attack_train11="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-TCP3_ALL.pkl"
    with open(Attack_train11,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDDoS15=len(data)

    Attack_train11="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-TCP4_ALL.pkl"
    with open(Attack_train11,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDDoS16=len(data)


    Attack_train12="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-UDP1_ALL.pkl"
    with open(Attack_train12,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDDoS17=len(data)


    Attack_train13="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-UDP2_ALL.pkl"
    with open(Attack_train13,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDDoS18=len(data)

    Attack_train13="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-UDP3_ALL.pkl"
    with open(Attack_train13,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDDoS19=len(data)

    Attack_train13="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-UDP4_ALL.pkl"
    with open(Attack_train13,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDDoS20=len(data)

    Attack_train13="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-UDP5_ALL.pkl"
    with open(Attack_train13,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDDoS21=len(data)

    Attack_train13="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-UDP6_ALL.pkl"
    with open(Attack_train13,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDDoS22=len(data)

    Attack_train13="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-UDP7_ALL.pkl"
    with open(Attack_train13,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDDoS23=len(data)

    Attack_train13="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DDoS-UDP8_ALL.pkl"
    with open(Attack_train13,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDDoS24=len(data)


    labels.extend([4]*(lenDDoS1+lenDDoS2+lenDDoS3+lenDDoS4+lenDDoS5+lenDDoS6+lenDDoS7+lenDDoS8+lenDDoS9+lenDDoS10+lenDDoS11+lenDDoS12+lenDDoS13+lenDDoS14+lenDDoS15+lenDDoS16+lenDDoS17+lenDDoS18+lenDDoS19+lenDDoS20+lenDDoS21+lenDDoS22+lenDDoS23+lenDDoS24)) 
    # CLASS DDoS


    Attack_train14="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-ICMP1_ALL.pkl"
    with open(Attack_train14,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDoS1=len(data)

    Attack_train14="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-ICMP2_ALL.pkl"
    with open(Attack_train14,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDoS2=len(data)

    Attack_train14="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-ICMP3_ALL.pkl"
    with open(Attack_train14,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDoS3=len(data)

    Attack_train14="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-ICMP4_ALL.pkl"
    with open(Attack_train14,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDoS4=len(data)


    Attack_train15="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-SYN1_ALL.pkl"
    with open(Attack_train15,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDoS5=len(data)

    Attack_train15="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-SYN2_ALL.pkl"
    with open(Attack_train15,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDoS6=len(data)

    Attack_train15="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-SYN3_ALL.pkl"
    with open(Attack_train15,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDoS7=len(data)

    Attack_train15="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-SYN4_ALL.pkl"
    with open(Attack_train15,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDoS8=len(data)

    Attack_train16="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-TCP1_ALL.pkl"
    with open(Attack_train16,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDoS9=len(data)

    Attack_train16="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-TCP2_ALL.pkl"
    with open(Attack_train16,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDoS10=len(data)

    Attack_train16="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-TCP3_ALL.pkl"
    with open(Attack_train16,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDoS11=len(data)

    Attack_train16="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-TCP4_ALL.pkl"
    with open(Attack_train16,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDoS12=len(data)


    Attack_train17="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-UDP1_ALL.pkl"
    with open(Attack_train17,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDoS13=len(data)

    Attack_train17="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-UDP2_ALL.pkl"
    with open(Attack_train17,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDoS14=len(data)

    Attack_train17="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-UDP3_ALL.pkl"
    with open(Attack_train17,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDoS15=len(data)

    Attack_train17="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/ALL_Fingerprints/nilsimsa_TCP-IP-DoS-UDP4_ALL.pkl"
    with open(Attack_train17,'rb') as f:
            data=pickle.load(f)
    train.extend(data)

    lenDoS16=len(data)

    labels.extend([5]*(lenDoS1+lenDoS2+lenDoS3+lenDoS4+lenDoS5+lenDoS6+lenDoS7+lenDoS8+lenDoS9+lenDoS10+lenDoS11+lenDoS12+lenDoS13+lenDoS14+lenDoS15+lenDoS16)) 
    # CLASS DoS


    Normal_train="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/normal/train/5s/nilsimsa_normal_ALL.pkl"
    with open(Normal_train,'rb') as f:
            data=pickle.load(f)
    train.extend(data)
        
    labels.extend([0]*len(data))

    savepath="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/train_categorical.pkl"
    with open(savepath,'wb') as f:
        pickle.dump(train, f)
    savepath="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/label_train_categorical.pkl"
    with open(savepath,'wb') as f:
        pickle.dump(labels, f)
    return print("saved")


def joindatasets_categorical():
    dataset_categorical=[]
    labels_categorical=[]

    path_test_categorical="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/test_categorical.pkl"
    with open(path_test_categorical,'rb') as f:
           data=pickle.load(f)
    dataset_categorical.extend(data)

    path_train_categorical="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/train_categorical.pkl"
    with open(path_train_categorical,'rb') as f:
           data=pickle.load(f)
    dataset_categorical.extend(data)

    path_label_test_categorical="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/label_test_categorical.pkl"
    with open(path_label_test_categorical,'rb') as f:
           data=pickle.load(f)
    labels_categorical.extend(data)

    path_label_train_categorical="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/label_train_categorical.pkl"
    with open(path_label_train_categorical,'rb') as f:
           data=pickle.load(f)
    labels_categorical.extend(data)

    savepath="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/dataset_categorical.pkl"
    with open(savepath,'wb') as f:
        pickle.dump(dataset_categorical, f)
    savepath="/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/labels_categorical.pkl"
    with open(savepath,'wb') as f:
        pickle.dump(labels_categorical, f)
    return print("saved")

categoricalDataset_TEST()
categoricalDataset_Train()
joindatasets_categorical()
