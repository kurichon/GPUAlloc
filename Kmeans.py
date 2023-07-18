from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
print(os.getcwd())
data = pd.read_csv("./data/CNN.csv")
data.columns = ["GPU Util%","GPU Memory"]
data['row_number'] = data.reset_index().index

for i in range(len(data.columns)):
    print (len(data.columns))
data_to_cluster = pd.DataFrame(data["GPU Memory"])
#data_to_cluster = pd.DataFrame(data["GPU Util%"])

kmeans = KMeans(n_clusters = 2)
#kmeans.fit(data_to_cluster)
#predict = kmeans.predict(data_to_cluster)

print(data_to_cluster)
#data_to_cluster['label']= kmeans.fit_predict(data[['GPU Util%']])
data_to_cluster['label']= kmeans.fit_predict(data[['GPU Memory']])
#clustered_data = data_to_cluster.assign(label=kmeans.fit_predict(data[['GPU Util%']]))
#data_to_cluster["cluster"] = predict
#data_to_cluster = data.sort_values("cluster")
print(data_to_cluster)

figure,axes = plt.subplots()
#data_to_cluster[data_to_cluster['label']==0].plot.scatter(x='GPU Util%',y='label', s=50,color='white',edgecolor='black',ax = axes)
#data_to_cluster[data_to_cluster['label']==1].plot.scatter(x='GPU Util%',y='label', s=50,color='white',edgecolor='red',ax = axes)

data_to_cluster[data_to_cluster['label']==0].plot.scatter(x='GPU Memory',y='label', s=50,color='white',edgecolor='black',ax = axes)
data_to_cluster[data_to_cluster['label']==1].plot.scatter(x='GPU Memory',y='label', s=50,color='white',edgecolor='red',ax = axes)
plt.scatter(kmeans.cluster_centers_.ravel(),[0.5]*len(kmeans.cluster_centers_),s=100,color='green',marker='*')




plt.show()
