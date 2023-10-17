import pandas as pd
import os
import matplotlib.pyplot as plt
import re


#optimize constants make constant.py
#create representation per model






file_list = []
dfs = []
#os.listdir('dir_path'): Return the list of files and directories in a specified directory path.

hp_model = ["vgg11","vgg16","vgg19","googlenet","lenet","densenet40-k12","densenet100-k12","inception3","inception4","resnet20","resnet50","resnet101"]
#filename =  "/data_sample.csv"
#path = 'data/GTX1080_imagenet/'
#path = 'data/GTX1080_cifar10/'
path = 'data/TitanX_cifar10/'
#path = 'data/GTX1080_cifar10/'
#path = 'data/GTX1080_cifar10/'
list_of_models = os.listdir(path)
for models in hp_model:
    r = re.compile(".*"+ models +".*")
    
    file_list = list(filter(r.match, list_of_models))

    for csv in file_list:
        df = pd.read_csv(path + csv)
        dfs.append(df)
#df = pd.read_csv(os.getcwd() + filename)
#print(len(df))

final_df = pd.concat(dfs,ignore_index=True)
print(final_df)
headers = list(final_df)
num_of_headers = len(headers)
num_of_chunks = 5

for i in range(0,num_of_headers):
    #chunks = pd.read_csv(os.getcwd() + ,chunksize=round(len(df)/num_of_chunks))
    #for j in range(0,num_of_chunks):
     #   df_graph = pd.DataFrame(next(chunks))
    x = final_df.index
    y = final_df.iloc[:,i]
    #linewidth = 2
    
    plt.plot(x,y,linewidth=0.5)
    plt.title(headers[i])
        #plt.show() if you want to check smaller chunks
    plt.show()









#headers = ['GPU Util%','GPU Memory']
#data = pandas.read_csv(os.getcwd() + "/data/data_to_graph.csv")
#chunks = pandas.read_csv(os.getcwd() + "/data/data_to_graph.csv",chunksize=5_000)
#data_1 = next(chunks)
#data_2 = next(chunks)
#data_3 = next(chunks)
#data_4 = next(chunks)
