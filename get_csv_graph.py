import pandas as pd
import os
import matplotlib.pyplot as plt
import re
import numpy as np
import csv
#optimize constants make constant.py
#create representation per model

'''get_csv_graph.py converts the exec_time.csv to segregated dataframes depending on configuration and graphs the existing data'''




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
#for _epoch in hp_epoch:
#    for opt in range(len(gpu_opt_df)):
 #       gpu_opt_epoch_df.append(gpu_opt_df[opt].loc[df['Epoch'] == _epoch])
                
#print((gpu_opt_bsize_df))
#0-2 sgd 


'''Graph epoch data'''
x = []
y = []
for i in range(len(gpu_opt_epoch_df)):
    x.append(gpu_opt_epoch_df[i].index)
    y.append(gpu_opt_epoch_df[i].iloc[:,6])

fig1, ax = plt.subplots(3, 2,figsize=(3,8),layout='constrained')

fig1.supxlabel("Data Points")
fig1.supylabel("Seconds")

ax[0,0].set_title('GTX1080 10 epochs')
ax[0,1].set_title('GTX1080 20 epochs')
ax[1,0].set_title('RTX2070 10 epochs')
ax[1,1].set_title('RTX2070 20 epochs')
ax[2,0].set_title('TitanX 10 epochs')
ax[2,1].set_title('TitanX 20 epochs')

idx=0
for i in range(3):
    for j in range(2):
        if j == 1:
            idx+=1
        #print(idx)
        ax[i,j].plot(x[idx],y[idx],linewidth=0.5,color = 'b',label="SGD")
        ax[i,j].plot(x[idx+2],y[idx+2],linewidth=0.5,color = 'g',label="Adam")
        ax[i,j].set_ylim([0,33000])
        ax[i,j].legend()
    if idx < 7:
        idx+=3
        #print(idx)        



# #20
# ax[0,0].plot(x[0],y[0],linewidth=0.5,color = 'b',label="SGD")
# ax[0,0].plot(x[2],y[2],linewidth=0.5,color = 'g',label="Adam")
# ax[0,0].set_ylim([0,30000])
# ax[0,0].legend()

# #20
# ax[0,1].plot(x[1],y[1],linewidth=0.5,color = 'b',label="SGD")
# ax[0,1].plot(x[3],y[3],linewidth=0.5,color = 'g',label="Adam")
# ax[0,1].set_ylim([0,30000])
# ax[0,1].legend()

# ax[1,0].plot(x[4],y[4],linewidth=0.5,color = 'b',label="SGD")
# ax[1,0].plot(x[6],y[6],linewidth=0.5,color = 'g',label="Adam")
# ax[1,0].set_ylim([0,30000])
# ax[1,0].legend()

# ax[1,1].plot(x[5],y[5],linewidth=0.5,color = 'b',label="SGD")
# ax[1,1].plot(x[7],y[7],linewidth=0.5,color = 'g',label="Adam")
# ax[1,1].set_ylim([0,30000])
# ax[1,1].legend()


# ax[2,0].plot(x[8],y[8],linewidth=0.5,color = 'b',label="SGD")
# ax[2,0].plot(x[10],y[10],linewidth=0.5,color = 'g',label="Adam")
# ax[2,0].set_ylim([0,30000])
# ax[2,0].legend()

# ax[2,1].plot(x[9],y[9],linewidth=0.5,color = 'b',label="SGD")
# ax[2,1].plot(x[11],y[11],linewidth=0.5,color = 'g',label="Adam")
# ax[2,1].set_ylim([0,30000])
# ax[2,1].legend()

plt.show()
plt.close()

'''Graph Batch Size Data'''
        
x = []
y = []    
    
for i in range(len(gpu_opt_bsize_df)):
    x.append(gpu_opt_bsize_df[i].index)        
    y.append(gpu_opt_bsize_df[i].iloc[:,6])
   
fig2, ax = plt.subplots(3, 4,figsize=(3,8),layout='constrained')
#10
#fig2.tight_layout()
fig2.supxlabel("Data Points")
fig2.supylabel("Seconds")
ax[0,0].set_title('GTX1080 Batch Size 8')
ax[0,1].set_title('GTX1080 Batch Size 16')
ax[0,2].set_title('GTX1080 Batch Size 32')
ax[0,3].set_title('GTX1080 Batch Size 64')
ax[1,0].set_title('RTX2070 Batch Size 8')
ax[1,1].set_title('RTX2070 Batch Size 16')
ax[1,2].set_title('RTX2070 Batch Size 32')
ax[1,3].set_title('RTX2070 Batch Size 64')
ax[2,0].set_title('TitanX Batch Size 8')
ax[2,1].set_title('TitanX Batch Size 16')
ax[2,2].set_title('TitanX Batch Size 32')
ax[2,3].set_title('TitanX Batch Size 64')

idx=0

# 0,4
# 1,5
# 2,6
# 3,7
# 8,12
# 9,13
# 10,14
# 11,15
# 16,20
# 17,21
# 18,22
# 19,23

for i in range(3):
    for j in range(4):
        #print(idx)
        ax[i,j].plot(x[idx],y[idx],linewidth=0.5,color = 'b',label="SGD")
        ax[i,j].plot(x[idx+4],y[idx+4],linewidth=0.5,color = 'g',label="Adam")
        ax[i,j].set_ylim([0,33000])
        ax[i,j].legend()
        idx+=1
    if idx < 16:
        idx+=4
        #print(idx)
plt.show()
plt.close()


'''Compare the difference between Adam vs SGD, 10 vs 20 epoch, 8 vs 16 vs 32 vs 64 batch sizes'''

divided = []

print(gpu_opt_df)


for i in range(len(gpu_opt_df)):
    if i % 2 !=0:
        continue
    print(i)
    x = np.array(gpu_opt_df[i].iloc[:,6].tolist())
    y = np.array(gpu_opt_df[i+1].iloc[:,6].tolist())
    divided_array = np.divide(x,y) #sgd/adam
    divided.append(list(divided_array))
#print(divided)

plt.clf()
#fig2, ax = plt.subplots(1, 2,figsize=(3,8),layout='constrained')
#x.append([x for x in range(len(divided))]) 
#y.append(divided) 
#print(len(y[0]))
plt.plot(divided[0],color = 'b',label="GTX1080",linewidth=0.8)
plt.plot(divided[1],color = 'g',label="RTX2070",linewidth=0.8)
plt.plot(divided[2],color = 'r',label="TitanX",linewidth=0.8)#,color = 'b',label="SGD")
#for i in range(len(divided)):
#list_len = [len(i) for i in divided]
    
plt.plot([1]*max([len(i) for i in divided]),color='black',linestyle='dashed')
#ax[0,0].plot([x for x in range(len(divided))],divided,linewidth=0.5,color = 'b',label="SGD")
#ax[i,j].plot(x[idx+4],y[idx+4],linewidth=0.5,color = 'g',label="Adam")
plt.ylim([0,1.5])
plt.title("SGD vs Adam Percentage Differences for Execution Time")
plt.xlabel("Data Points")
plt.ylabel("Percentage Difference (%)")
plt.legend()
plt.show()


#df2 = x - y
#df2 = [item for item in x if item not in y]
#print(df2)


