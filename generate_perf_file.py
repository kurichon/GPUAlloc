from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import csv
import constants
#print(os.getcwd())
#data = pd.read_csv("/home/ml-server/Chad/Code/GitHub/data/RESNET50.csv")
#data = pd.read_csv("./data/CNN.csv")
#data['row_number'] = data.reset_index().index
"""data is pandas dataframe"""
def generate_parameters(data,gpu,codename):
    codename = codename.split('_')
    perf_list_to_return = []
    for column in data:
        if column == "tx" or column == "rx":
            continue
        else:
            data_to_cluster = pd.DataFrame(data[column])
            kmeans = KMeans(n_clusters = 2,n_init=10)
            data_to_cluster['label']= kmeans.fit_predict(data[[column]])
        
            data_to_cluster = data_to_cluster.sort_values(by=column)
        
            #figure,axes = plt.subplots()
            #data_to_cluster[data_to_cluster['label']==0].plot.scatter(x=column,y='label', s=50,color='white',edgecolor='black',ax = axes)
            #data_to_cluster[data_to_cluster['label']==1].plot.scatter(x=column,y='label', s=50,color='white',edgecolor='red',ax = axes)
            #plt.scatter(kmeans.cluster_centers_.ravel(),[0.5]*len(kmeans.cluster_centers_),s=100,color='green',marker='*')
            #plt.show()
            
            burst_len = len(data_to_cluster.loc[data_to_cluster['label'] == 0])
            idle_len = len(data_to_cluster.loc[data_to_cluster['label'] == 1])
        
            if column == "GPU Util%":
                peak_len = len(data_to_cluster[(data_to_cluster[[column]]>90).all(axis=1)])
            elif column == "GPU Memory": #add GPU type later
                if gpu == "TitanX":                
                    peak_len = len(data_to_cluster[(data_to_cluster[[column]]>11059.2).all(axis=1)])
                else:
                    peak_len = len(data_to_cluster[(data_to_cluster[[column]]>7372.8).all(axis=1)])
            elif column == "GPU Mem Util%":
                peak_len = len(data_to_cluster[(data_to_cluster[[column]]>90).all(axis=1)])
            elif column == "CPU Util%":#,tx,rx
                peak_len = len(data_to_cluster[(data_to_cluster[[column]]>90).all(axis=1)])
            #print(burst/burst_len)
            #print(sum(float(idle))/idle_len)
           #     peak_len =
            avg_burst_time = burst_len/len(data)
            avg_idle_time = idle_len/len(data)
            avg_peak_consumption = peak_len/len(data)
            
            perf_list_to_return.append({constants.perf_file_header[0]:codename[0],constants.perf_file_header[1]:codename[1],
            constants.perf_file_header[2]:codename[2],constants.perf_file_header[3]:codename[3],constants.perf_file_header[4]:codename[4],
            constants.perf_file_header[5]:codename[5],constants.perf_file_header[6]:column,constants.perf_file_header[7]:avg_burst_time,
            constants.perf_file_header[8]:avg_idle_time,constants.perf_file_header[9]:avg_peak_consumption})# +',{0},{1},{2},{3}\n'.format(column,avg_burst_time,avg_idle_time,avg_peak_consumption))
    return perf_list_to_return

        #writer = writer.writerow(gpu_resource)
        #writer = writer.writerow({'GPU Util%': get_gpu_utilization(), 'GPU Memory': get_gpu_memory()})
        #writer = writer.writerow({'GPU Util%': gpu_resource[0], 'GPU Memory': gpu_resource[1],'CPU Util%': gpu_resource[2]})
       # print("Average Burst Time: {0}".format(burst_len/len(data)))
       # print("Average Idle Time: {0}".format(idle_len/len(data)))
       # print("Average Peak Consumption: {0}".format(peak_len/len(data)))
