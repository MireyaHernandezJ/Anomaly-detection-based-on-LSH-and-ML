import pickle
import os


#leer cantidad de FP por clase
# n_splits="20s" ### pps
# filename="petya" #petya, powerghost, wannacry, badrabbit, clean 
# directory = "/home/mireya/Documentos/Datasets-IoT/ICEdataset/ICEdataset_FPs_multiclass/hash_vectors_%s/%s_nilsimsa.pkl" % (n_splits,filename)
# with open(directory,'rb') as f:
#      data=pickle.load(f)

#print(len(data))

n_splits="20s" ### pps
directory = "/home/mireya/Documentos/Datasets-IoT/ICEdataset/ICEdataset_FPs_binary/labels_%s_nilsimsa.pkl" % (n_splits)
with open(directory,'rb') as f:
     data=pickle.load(f)
print(len(data))

def count_ceros(arr):
    count=0
    for num in arr:
        if num==0:
               count+=1
    return count 
print(count_ceros(data))
print(len(data)-count_ceros(data))
####################### CODE FOR PPS


# n_splits="600" ### pps

def joining(filename,nsplits,nfiles):
    pickles=[]
    for i in range(1,nfiles+1):
        filepath= "/home/mireya/Documentos/Datasets-IoT/ICEdataset/ICEdataset_FPs_pps_binary/%s/%s/nilsimsa_%s_%d.pkl" % (nsplits, filename,filename,i)
        savepath="/home/mireya/Documentos/Datasets-IoT/ICEdataset/ICEdataset_FPs_pps_binary/%s/nilsimsa_%s_ALL.pkl" % (nsplits,filename)
        with open(filepath,'rb') as f:
            data=pickle.load(f)
        pickles.extend(data) 
    with open(savepath,'wb') as f:
        pickle.dump(pickles, f)
    return print("joined")


#check for empty files
def check(filename,nsplits,nfiles):
  
    for i in range(1,nfiles+1):
        filepath= "/home/mireya/Documentos/Datasets-IoT/ICEdataset/ICEdataset_FPs_pps_binary/%s/%s/nilsimsa_%s_%d.pkl" % (nsplits, filename,filename,i)
        savepath="/home/mireya/Documentos/Datasets-IoT/ICEdataset/ICEdataset_FPs_pps_binary/%s/nilsimsa_%s_ALL.pkl" % (nsplits,filename)
        if not os.path.exists(filepath):
            print(f"File {filepath} {i} does not exist. Skipping...")
            continue
        try:
            with open(filepath,'rb') as f:
                data=pickle.load(f)
           
        
        except EOFError:
            print(f"EOFError: Ran out of input in file {filepath}. Skipping...")
            continue
    
    return print("checked")

# #PRIMERO HAY QUE UNIR DE CADA FILENAME Y LUEGO CREAR EL DATASET
# filename="clean" #petya, powerghost, wannacry, badrabbit, clean 
# directory = "/home/mireya/Documentos/Datasets-IoT/ICEdataset/ICEdataset_splits_pps/%s/%s" % (n_splits, filename)
# n_files = len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
# #print(n_files)
# #joining(filename=filename,nsplits=n_splits,nfiles=n_files)
# #check(filename=filename,nsplits=n_splits,nfiles=n_files)


def dataset_create(nsplits):
    dataset=[]
    Attack_petya="/home/mireya/Documentos/Datasets-IoT/ICEdataset/ICEdataset_FPs_pps_binary/%s/nilsimsa_petya_ALL.pkl"%(nsplits)
    with open(Attack_petya,'rb') as f:
            data=pickle.load(f)
    dataset.extend(data)

    Attack_powerghost="/home/mireya/Documentos/Datasets-IoT/ICEdataset/ICEdataset_FPs_pps_binary/%s/nilsimsa_powerghost_ALL.pkl"%(nsplits)
    with open(Attack_powerghost,'rb') as f:
            data=pickle.load(f)
    dataset.extend(data)

    Attack_wannacry="/home/mireya/Documentos/Datasets-IoT/ICEdataset/ICEdataset_FPs_pps_binary/%s/nilsimsa_wannacry_ALL.pkl"%(nsplits)
    with open(Attack_wannacry,'rb') as f:
            data=pickle.load(f)
    dataset.extend(data)

    Attack_badrabbit="/home/mireya/Documentos/Datasets-IoT/ICEdataset/ICEdataset_FPs_pps_binary/%s/nilsimsa_badrabbit_ALL.pkl"%(nsplits)
    with open(Attack_badrabbit,'rb') as f:
            data=pickle.load(f)
    dataset.extend(data)

    labels=[1]*len(dataset)

    
    Normal="/home/mireya/Documentos/Datasets-IoT/ICEdataset/ICEdataset_FPs_pps_binary/%s/nilsimsa_clean_ALL.pkl"%(nsplits)
    with open(Normal,'rb') as f:
            data=pickle.load(f)
    dataset.extend(data)
        
    label_normal=[0]*(len(dataset)-len(labels))    
    labels.extend(label_normal)

    savepath="/home/mireya/Documentos/Datasets-IoT/ICEdataset/ICEdataset_FPs_pps_binary/%s/dataset_%s.pkl"%(nsplits,nsplits)
    with open(savepath,'wb') as f:
        pickle.dump(dataset, f)
    savepath="/home/mireya/Documentos/Datasets-IoT/ICEdataset/ICEdataset_FPs_pps_binary/%s/labels_%s.pkl"%(nsplits,nsplits)
    with open(savepath,'wb') as f:
        pickle.dump(labels, f)
    return print("saved")

# dataset_create(n_splits)
 
# check_dataset="/home/mireya/Documentos/Datasets-IoT/ICEdataset/ICEdataset_FPs_pps_binary/%s/dataset_%s.pkl"%(n_splits,n_splits)
# with open(check_dataset,'rb') as f:
#         data=pickle.load(f)
# print(len(data))
# print(data[0])
# print(data[100])


# check_labels="/home/mireya/Documentos/Datasets-IoT/ICEdataset/ICEdataset_FPs_pps_binary/%s/labels_%s.pkl"%(n_splits,n_splits)
# with open(check_labels,'rb') as f:
#         data=pickle.load(f)
# print(len(data))
# print(data[0])
# print(data[100])        
