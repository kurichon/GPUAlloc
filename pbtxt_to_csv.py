# import necessary libraries
import pandas as pd
import os
import glob
from google.protobuf import text_format
from collections import Counter
import tensorflow as tf
  
# use glob to get all the csv files 
# in the folder
path = os.getcwd() + "/graphs/"
pbtxt_files = glob.glob(os.path.join(path, "*.txt"))
  
def open_pbtxt(file):
    with open(file) as f:
        pbtxt = f.read()
        graph_def = text_format.Parse(pbtxt, tf.compat.v1.GraphDef(), allow_field_number=1)
        return graph_def
def pbtxt_extract_op_name(p):
    graph_node_list = []
    for node in p.node:
        if "GPU" in node.device:
            graph_node_list.append(node)
    op_name_list_in_pbtxt = [node.op for node in graph_node_list]
    return op_name_list_in_pbtxt
    
for f in pbtxt_files:
    ops = []
    # read the csv file
    # = pd.read_csv(f)
      
    # print the location and filename
    print('Location:', f)
    print('File Name:', f.split("\\")[-1])
    pb_file = open_pbtxt(f)
    ops += pbtxt_extract_op_name(pb_file)
    # print the content
    #print('Content:')
    #display(df)
    #print()
print(ops)
print(len(ops))  
unique_ops = pd.Series(ops).drop_duplicates().tolist()
op_count = Counter(ops)
print (op_count)

node_frequency = ''

for x in unique_ops:
    freq = op_count[x]/len(ops)
    node_frequency+= '{0},{1}\n'.format(x,freq)
with open(datetime.now()+"_nodefrequency.csv", 'w+') as f:
    f.writelines(node_frequency)

print(node_freq)
    
#get frequency

# loop over the list of csv files
# for f in pbtxt_files:
      
    # # read the csv file
    # df = pd.read_csv(f)
      
    # # print the location and filename
    # print('Location:', f)
    # print('File Name:', f.split("\\")[-1])
      
    # # print the content
    # print('Content:')
    # display(df)
    # print()
