import pandas as pd
import os
import matplotlib.pyplot as plt
import re
from datetime import datetime


#optimize constants make constant.py
#create representation per model






file_list = []
dfs = []
#os.listdir('dir_path'): Return the list of files and directories in a specified directory path.

hp_model = ["vgg11","vgg16","vgg19","googlenet","_lenet","densenet40-k12","densenet100-k12","inception3","inception4","resnet20","resnet50","resnet101"]
exec_time = []
#filename =  "/data_sample.csv"
path = 'data/Real Data'
#path = 'data/GTX1080_cifar10/'
#path = 'data/TitanX_cifar10/'
#path = 'data/GTX1080_cifar10/'
#path = 'data/GTX1080_cifar10/'
#list_of_logfiles = os.listdir(path)
list_of_logfiles = ''

for subdir, dirs, files in os.walk(path):
    for file in files:
        if "_logfile.txt" in file:
            list_of_logfiles += os.path.join(subdir,file) + '\n'
list_of_logfiles = list(filter(None,list_of_logfiles.split('\n')))
print(list_of_logfiles)
for logfile in list_of_logfiles:
    with open(logfile,'r') as _file:
        #for model in hp_model:
            #start = 0
            #end = 0                
            for line in _file:
                for model in hp_model:
                    if model in line:
                        print(model)
                        count +1
                        count +1
                        if count = 2
                            get exec time
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









#headers = ['GPU Util%','GPU Memory']
#data = pandas.read_csv(os.getcwd() + "/data/data_to_graph.csv")
#chunks = pandas.read_csv(os.getcwd() + "/data/data_to_graph.csv",chunksize=5_000)
#data_1 = next(chunks)
#data_2 = next(chunks)
#data_3 = next(chunks)
#data_4 = next(chunks)
