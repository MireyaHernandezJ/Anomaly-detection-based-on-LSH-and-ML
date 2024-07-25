from scapy.all import *
from nilsimsa import *
import pickle
import os

def readPCAPfiles(filename,nsplits,nfiles):

    #vectors_hash_Nilsimsa=[]

    #for i in range(1,nfiles+1):
    for i in range(2720,2721):
        vectors_hash_Nilsimsa=[]

        file=rdpcap("/home/mireya/Documentos/Datasets-IoT/ICEdataset/ICEdataset_splits_pps/%s/%s/%d.pcap" % (nsplits, filename,i))
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
        
        with open("/home/mireya/Documentos/Datasets-IoT/ICEdataset/ICEdataset_FPs_pps_binary/%s/%s/nilsimsa_%s_%d.pkl" % (nsplits, filename,filename,i),'wb') as f:
            pickle.dump(vectors_hash_Nilsimsa,f)
  

n_splits="600" ### pps: 100
filename="clean" #petya, powerghost, wannacry, badrabbit, clean 
directory = "/home/mireya/Documentos/Datasets-IoT/ICEdataset/ICEdataset_splits_pps/%s/%s" % (n_splits, filename)
n_files = len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
print(n_files)
readPCAPfiles(filename=filename, nsplits=n_splits,nfiles=n_files)
