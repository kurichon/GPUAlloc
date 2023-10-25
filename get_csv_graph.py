import pandas as pd
import os
import matplotlib.pyplot as plt
import re

import csv
#optimize constants make constant.py
#create representation per model






hyperparam_header = ['GPU', 'Model','Batch Size','Optimizer','Epoch','Dataset','Execution Time']
file_list = []
dfs = []
#os.listdir('dir_path'): Return the list of files and directories in a specified directory path.

hp_model = ["vgg11","vgg16","vgg19","googlenet","_lenet","densenet40-k12","densenet100-k12","inception3","inception4","resnet20","resnet50","resnet101"]

gpu_model = ["GTX1080","RTX2070","TitanX"]
hp_batch_size = [8,16,32,64]
hp_optimizer = ['sgd','adam']
hp_epoch = [10,20]
hp_dataset = ["cifar10","imagenet"]

#filename =  "/data_sample.csv"

#path = 'data/GTX1080_imagenet/'
#path = 'data/GTX1080_cifar10/'
#path = 'data/TitanX_cifar10/'
#path = 'data/Real Data/RTX2070_cifar10/'
path = './exec_time.csv'
df = pd.read_csv(path)
                                                                                                                                                                       
gpu_df = [] #gtx1080,rtx2070,titanx
gpu_opt_df = [] #GPU , #sgd,adam,  0-1 , 2-3, 4-5
gpu_opt_bsize_df=[] #0-8, 9-15, 16-23
gpu_opt_epoch_df=[] #0-3, 4-7,8-11
for gpu in gpu_model:
    gpu_df.append(df.loc[df['GPU'] == gpu])
    for opt in hp_optimizer:
        gpu_opt_df.append(gpu_df[len(gpu_df)-1].loc[df['Optimizer'] == opt])
        for bsize in hp_batch_size:
            gpu_opt_bsize_df.append(gpu_opt_df[len(gpu_opt_df)-1].loc[df['Batch Size'] == bsize])
        for _epoch in hp_epoch:
            gpu_opt_epoch_df.append(gpu_opt_df[len(gpu_opt_df)-1].loc[df['Epoch'] == _epoch])
                
print(gpu_opt_epoch_df)
plot = gpu_opt_df[0].groupby("Execution Time").plot(kind="box",title="DataFrameGroupBy Plot")
    # #linewidth = 2float(t_temp.strftime("%H%M%S"))
#plt.plot(x,y,linewidth=0.5)
#plt.title("Knees")
        # #plt.show() if you want to check smaller chunks
#plt.show()
#plt.close()








#path = 'data/GTX1080_cifar10/'
# list_of_models = os.listdir(path)
# for models in hp_model:
    # r = re.compile(".*"+ models +".*")
    
    # file_list = list(filter(r.match, list_of_models))

    # for csv in file_list:
        # df = pd.read_csv(path + csv)
        # dfs.append(df)
# #df = pd.read_csv(os.getcwd() + filename)
# #print(len(df))

# final_df = pd.concat(dfs,ignore_index=True)
# print(final_df)
# headers = list(final_df)
# num_of_headers = len(headers)
# num_of_chunks = 5

# for i in range(0,num_of_headers):
    # #chunks = pd.read_csv(os.getcwd() + ,chunksize=round(len(df)/num_of_chunks))
    # #for j in range(0,num_of_chunks):
     # #   df_graph = pd.DataFrame(next(chunks))
    # x = final_df.index
    # y = final_df.iloc[:,i]
    # #linewidth = 2
    
    # plt.plot(x,y,linewidth=0.5)
    # plt.title(headers[i])
        # #plt.show() if you want to check smaller chunks
    # plt.show()









#headers = ['GPU Util%','GPU Memory']
#data = pandas.read_csv(os.getcwd() + "/data/data_to_graph.csv")
#chunks = pandas.read_csv(os.getcwd() + "/data/data_to_graph.csv",chunksize=5_000)
#data_1 = next(chunks)
#data_2 = next(chunks)
#data_3 = next(chunks)
#data_4 = next(chunks)
