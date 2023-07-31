import pandas
import os
df = pandas.read_csv(os.getcwd() + "/data/data_to_graph.csv")
print(df)

import pandas as pd
import matplotlib.pyplot as plt



headers = ['Time','GPU Util%','GPU Memory']

#data = pandas.read_csv(os.getcwd() + "/data/data_to_graph.csv")
#chunks = pandas.read_csv(os.getcwd() + "/data/data_to_graph.csv",chunksize=5_000)
#data_1 = next(chunks)
#data_2 = next(chunks)
#data_3 = next(chunks)
#data_4 = next(chunks)
for i in range(1,3):
    chunks = pandas.read_csv(os.getcwd() + "/data/data_to_graph2.csv",chunksize=30_000)
    for j in range(1,5):
        df = pd.DataFrame(next(chunks))
        x = df.index
        y = df.iloc[:,i]
    #linewidth = 2
    
        plt.plot(x,y,linewidth=0.5)
    if(i == 1):
        plt.title('GPU Util%')
    else:
        plt.title('GPU Memory')
    plt.show()
    
