import pandas as pd
import os
import matplotlib.pyplot as plt



filename =  "/data_sample.csv"
df = pd.read_csv(os.getcwd() + filename)
print(len(df))

headers = list(df)
num_of_headers = len(headers)
num_of_chunks = 5




#headers = ['GPU Util%','GPU Memory']
#data = pandas.read_csv(os.getcwd() + "/data/data_to_graph.csv")
#chunks = pandas.read_csv(os.getcwd() + "/data/data_to_graph.csv",chunksize=5_000)
#data_1 = next(chunks)
#data_2 = next(chunks)
#data_3 = next(chunks)
#data_4 = next(chunks)



for i in range(0,num_of_headers):
    chunks = pd.read_csv(os.getcwd() + filename,chunksize=round(len(df)/num_of_chunks))
    for j in range(0,num_of_chunks):
        df_graph = pd.DataFrame(next(chunks))
        x = df_graph.index
        y = df_graph.iloc[:,i]
    #linewidth = 2
    
        plt.plot(x,y,linewidth=0.5)
    plt.title(headers[i])
        #plt.show() if you want to check smaller chunks
    plt.show()
