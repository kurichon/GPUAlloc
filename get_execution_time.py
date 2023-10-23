import pandas as pd
import os
import matplotlib.pyplot as plt
import re
from datetime import datetime
import csv

from tqdm import tqdm

#optimize constants make constant.py
#create representation per model

exec_time_filename = "exec_time.csv"

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
exec_time = []
#filename =  "/data_sample.csv"
#path = 'data/Real Data/GTX1080_cifar10/'
#path = 'data/GTX1080_cifar10/'
#path = 'data/TitanX_cifar10/'
#path = 'data/GTX1080_cifar10/'
#path = 'data/GTX1080_cifar10/'
#list_of_logfiles = os.listdir(path)
list_of_logfiles = ''
for gpu in gpu_model:
    for dataset in hp_dataset:        
        path = './data/Real Data/{0}_{1}/'.format(gpu,dataset) 
        for subdir, dirs, files in os.walk(path):
            for file in files:
                if "_logfile.txt" in file:
                    list_of_logfiles += os.path.join(subdir,file) + '\n'
list_of_logfiles = list(filter(None,list_of_logfiles.split('\n')))
print(list_of_logfiles)

#count = 0
#found_exec = False

#for logfile in list_of_logfiles:
    #with open(logfile,'r') as _file:
        # #for model in hp_model:
            # #start = 0
            # #end = 0                
            # for line in _file:
                # for model in hp_model:
                    # if model in line:
                        # if found_exec == True:
                            # found_exec = False
                        # else:
                            # print(line)
                        
                    # if "Execution Time" in line:
                        # found_exec = True
                        #print(line.split(': ')[1].replace('\n',""))
                        #break
                        #   get model line and exec line to get the exec time for each configuration types
            #        if start == 0:
                  #      start = datetime.strptime(line.split('at ')[1].replace('\n',''), '%Y-%m-%d %H:%M:%S.%f')
                        
              #      else:
                  #      end = datetime.strptime(line.split('at ')[1].replace('\n',''), '%Y-%m-%d %H:%M:%S.%f')
                        
               # if start != 0 and end !=0:  
               #     print(end - start)
               #     start = 0
               #     end = 0        
#df = pd.read_csv(os.getcwd() + filename)
#print(len(df))

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

list_exec_time = []

for gpu in gpu_model:
        for _dset in hp_dataset:
            #if _dset == "imagenet":
            #    models_to_load = hp_model_imagenet
            #elif _dset == "cifar10":
            #    models_to_load = hp_model_cifar
            for model in tqdm(hp_model,desc='models'):
                for bsize in hp_batch_size:
                    for _epoch in hp_epoch:
                        for opt in hp_optimizer:
                            for logfile in list_of_logfiles:
                                with open(logfile,'r') as _file: 
                                    for line in _file:  
                                        
                                        if "{0}_{1}_{2}_{3}_{4} has Finished at ".format(
                                        gpu, model, bsize, opt, _epoch)in line:
                                            
                                            dataset = _file.name.split("_")[1].split("/")[0]
                                            exec_time = next(_file).split(": ")[1].replace("\n","")
                                            
                                            list_exec_time.append({hyperparam_header[0]: gpu, hyperparam_header[1]: model, hyperparam_header[2]: bsize,
                                            hyperparam_header[3]: opt, hyperparam_header[4]: _epoch, hyperparam_header[5]: dataset ,hyperparam_header[6] : exec_time})

                                        #update to csv format
    
print(list_exec_time)
with open(exec_time_filename + '.csv', 'w+') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=hyperparam_header)
    writer.writeheader()
    writer.writerows(list_exec_time)




#headers = ['GPU Util%','GPU Memory']
#data = pandas.read_csv(os.getcwd() + "/data/data_to_graph.csv")
#chunks = pandas.read_csv(os.getcwd() + "/data/data_to_graph.csv",chunksize=5_000)
#data_1 = next(chunks)
#data_2 = next(chunks)
#data_3 = next(chunks)
#data_4 = next(chunks)
