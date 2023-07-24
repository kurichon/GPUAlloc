from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
#print(os.getcwd())
data = pd.read_csv("/home/ml-server/Chad/Code/GitHub/data/RESNET50.csv")
#data = pd.read_csv("./data/CNN.csv")
data.columns = ["GPU Util%","GPU Memory"]
#data['row_number'] = data.reset_index().index

for column in data:
    data_to_cluster = pd.DataFrame(data[column])
    #data_to_cluster = pd.DataFrame(data["GPU Memory"])
    #data_to_cluster = pd.DataFrame(data["GPU Util%"])
    kmeans = KMeans(n_clusters = 2)
    #data_to_cluster['label']= kmeans.fit_predict(data[['GPU Util%']])
    data_to_cluster['label']= kmeans.fit_predict(data[[column]])

    #sort after labeling
    data_to_cluster = data_to_cluster.sort_values(by=column)
    #print(data_to_cluster)

    figure,axes = plt.subplots()
    #data_to_cluster[data_to_cluster['label']==0].plot.scatter(x='GPU Util%',y='label', s=50,color='white',edgecolor='black',ax = axes)
    #data_to_cluster[data_to_cluster['label']==1].plot.scatter(x='GPU Util%',y='label', s=50,color='white',edgecolor='red',ax = axes)

    data_to_cluster[data_to_cluster['label']==0].plot.scatter(x=column,y='label', s=50,color='white',edgecolor='black',ax = axes)
    data_to_cluster[data_to_cluster['label']==1].plot.scatter(x=column,y='label', s=50,color='white',edgecolor='red',ax = axes)
    plt.scatter(kmeans.cluster_centers_.ravel(),[0.5]*len(kmeans.cluster_centers_),s=100,color='green',marker='*')
    plt.show()
    # Average Resource Consumption
    #burst = data_to_cluster.loc[data_to_cluster['label'] == 0]
    #burst = data_to_cluster.loc[data_to_cluster['label'] == 0].sum()
    #idle = data_to_cluster.loc[data_to_cluster['label'] == 1]
    burst_len = len(data_to_cluster.loc[data_to_cluster['label'] == 0])
    idle_len = len(data_to_cluster.loc[data_to_cluster['label'] == 1])

    if column == "GPU Util%":
        peak_len = len(data_to_cluster[(data_to_cluster[[column]]>90).all(axis=1)])
    elif column == "GPU Memory": #add GPU type later
        peak_len = len(data_to_cluster[(data_to_cluster[[column]]>7372.8).all(axis=1)])
    #print(burst/burst_len)
    #print(sum(float(idle))/idle_len)
   #     peak_len =


    print("Average Burst Time: {0}".format(burst_len/len(data)))
    print("Average Idle Time: {0}".format(idle_len/len(data)))
    print("Average Peak Consumption: {0}".format(peak_len/len(data)))