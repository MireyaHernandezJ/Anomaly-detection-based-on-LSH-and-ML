import pickle

# #load pickle file
# pickle_file = open(
#     "/home/mireya/Documentos/Datasets-IoT/CICIoMT2024/FPs/splits_Wifi-MQTT/attack/train/5s/TCP-IP-DoS-UDP1/nilsimsa_TCP-IP-DoS-UDP1_40.pkl","rb")
# file_pkl = pickle.load(pickle_file)
# pickle_file.close()

# print(len(file_pkl))
# print(file_pkl[0])


#load pickle file
pickle_file = open(
    "/home/mireya/Descargas/nilsimsa_TCP-IP-DoS-TCP3_105.pkl","rb")
file_pkl = pickle.load(pickle_file)
pickle_file.close()

print(len(file_pkl))
print(file_pkl[0])
