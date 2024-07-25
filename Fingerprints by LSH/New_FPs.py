from scapy.all import *
from nilsimsa import *
import pickle
import os

# FOR NORMAL PCAP FILES FROM WIFI-MQTT dataset
def readPCAPfiles(scenario,filename,nsplits,nfiles,subsetsave):

    #vectors_hash_Nilsimsa=[]

    for i in range(1,nfiles+1):
        vectors_hash_Nilsimsa=[]

        file=rdpcap("/home/mireya/Documentos/%ss/%s/%s/%d.pcap"%(nsplits,scenario,subsetsave,i))
        raw_data=b''
        #read raw data from each packets of pcap file
        for packet in file:
            stream=raw(packet)
            #concatenate the raw data to obtain a big raw data
            raw_data+=stream

        # #Nilsimsa
        create_hash=Nilsimsa(raw_data).hexdigest()
        
        #convert hash into a vector of integers
        hash_nilsimsa=[int(create_hash[i:i+2],16)for i in range(0,len(create_hash),2)]
        
        #save FP
        vectors_hash_Nilsimsa.append(hash_nilsimsa)
        
        with open("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/%s/%s/%s/%ss/nilsimsa_%d.pkl"%(filename,scenario,subsetsave,nsplits,i),'wb') as f:
            pickle.dump(vectors_hash_Nilsimsa,f)
  
    # with open("/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/%s/%s/%s/%ss/nilsimsa_%s_%d.pkl"%(filename,scenario,subsetsave,nsplits,subset,i),'wb') as f:
    #     pickle.dump(vectors_hash_Nilsimsa,f)



n_splits="5" #5,10,15,20
filename="splits_Wifi-MQTT" #splits_Bluetooth or splits_Wifi-MQTT
scenario="normal" # normal 
subsetsave="train" # test or train , where to save the FPs
directory = "/home/mireya/Documentos/%ss/%s/%s" % (n_splits, scenario, subsetsave)
n_files = len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
print(n_files)
readPCAPfiles(scenario=scenario, filename=filename, nsplits=n_splits, nfiles=n_files,subsetsave=subsetsave)


