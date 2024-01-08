import os
import torch
import random
import sys
import itertools
import numpy as np
import csv
import pandas as pd
from tqdm import tqdm
import networkx as nx
import build_graph
from models.pytorch.gnn_framework import GNN
from types import SimpleNamespace
import get_JCT
import argparse
import matplotlib.pyplot as plt
from models.pytorch.gcn.layer import GCNLayer

csv_header = ["Model","GPU","Exec Time","Efficiency"]


"""Loading Prediction Model"""
gnn_args=dict(nfeat=3,
            nhid=64,
            nodes_out=None,
            graph_out=16,#6
            dropout=0.0,
            device='cuda',
            first_conv_descr=dict(layer_type=GCNLayer, args=dict()),
            middle_conv_descr=dict(layer_type=GCNLayer, args=dict()),
            fc_layers=3,
            conv_layers= lambda adj: adj.shape[1] // 2,
            skip=False,
            gru=True,
            fixed=True,
            variable=True,
            transfer=True)

#pre_trained=pre-train_GTX1080_3.pkl


def move_cuda():
    if features['val'][0].is_cuda:
        # already on CUDA
        return
    features['val'] = [x.cuda() for x in features['val']]
    adj['val'] = [x.cuda() for x in adj['val']]
    #graph_labels['val'] = [x.cuda() for x in graph_labels[dset]]

def check_if_file_exists(path):
    try:
        is_existing=True
        with open(path) as f:
            f.close()
    except FileNotFoundError:
        # file doesn't exist
        is_existing= False
        pass

    return is_existing




def find_model_list(pbtxt_path):
    pbtxt_files = []
    pbtxt_files.append([file_name[:-6] for file_name in os.listdir(pbtxt_path)])
    model_list = list(itertools.chain(*pbtxt_files))

    return model_list


def load_dataset(data_path):
    with open(data_path, 'rb') as f:
        print(len(data_path))
        (adj, features) = torch.load(f)
    return adj,features


"""
Getting the Input:
Conversion to Adjacent and Feature matrices"""

pbtxt_path = "./data/eval_graphs/graphs/"
nodetype_file = "./data/eval_graphs/node-frequency.csv"

model_list = find_model_list(pbtxt_path)
model_graph_label_dict = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),(0, 0), (0, 0), (0, 0), (0, 0)]

print(model_list)
n_graphs = {'val': [1] * len(model_list)}
adj = {}
features = {}
#graph_labels = {}

n_of_groups = 100 #default

set_adj = [[] for _ in n_graphs['val']]
set_features = [[] for _ in n_graphs['val']]
#set_graph_labels = [[] for _ in n_graphs['val']]

t = tqdm(total=len(model_list), desc="input graphs", leave=True, unit=' graphs')

for idx_i,models in enumerate(model_list):
    model_name = models #codename
    print(model_name)
    g = build_graph.build_nx_graph(pbtxt_path, nodetype_file,model_name)
    g.remove_nodes_from(list(nx.isolates(g)))
    #nx.draw(g)
    #plt.show()
    if nx.number_connected_components(g) > 1:
        g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
    #new with gan
    #dnn_name = model_name.split('_')[1]
    dnn_name = model_name
    
    #for smaller networks (few operations)
    if dnn_name == "cvae" or dnn_name == "gan" or dnn_name == "dcgan" or dnn_name == "conditionalgan" or dnn_name == "anomalydetection":
        grouped_g = build_graph.group_nx_graph(g, 18)
    else:
        grouped_g = build_graph.group_nx_graph(g, n_of_groups)
    #grouped_g = build_graph.group_nx_graph(g, n_of_groups)
    grouped_g.remove_edges_from(nx.selfloop_edges(grouped_g))
    #nx.draw_networkx(grouped_g)
    #plt.show()
    nodes = list(grouped_g)
    adj_temp = nx.to_numpy_array(grouped_g, nodes)
    t.update(1)

    f1 = np.array([grouped_g.nodes[i]['tensorsize'] for i in range(grouped_g.number_of_nodes())])
    f2 = np.array([grouped_g.nodes[i]['n_of_grouped_nodes'] for i in range(grouped_g.number_of_nodes())])
    f3 = np.array([grouped_g.nodes[i]['node_type'] for i in range(grouped_g.number_of_nodes())])
    features_temp = np.stack([f1, f2, f3], axis=1)
    #graph_labels_temp = np.asarray(model_graph_label_dict).flatten()
    
    set_adj[idx_i].append(adj_temp)
    set_features[idx_i].append(features_temp)
    #set_graph_labels[i].append(graph_labels_temp)
    #break
#break
t.close()

adj['val'] = [torch.from_numpy(np.asarray(adjs)).float() for adjs in set_adj]
features['val'] = [torch.from_numpy(np.asarray(fs)).float() for fs in set_features]
#graph_labels['val'] = [torch.from_numpy(np.asarray(gls)).float() for gls in set_graph_labels]

#print(features['val'])
#print(len(features['val']))
#print(len(features['val'][0]))
#print(len(features['val'][0][0]))
#print(len(features['val'][0][0][0]))
#print(adj)
#print(features)
#print(graph_labels)



resource = ["Mem_Util%","GPU_util","CPU_Util%","GPU_Memory"]
value_category = ["avg Active time","avg Idle time", "avg Peak consumption","Execution Time"]
trained_model_name = ["GTX1080_with_TL.pkl","RTX2070_with_TL.pkl","TitanX_with_TL.pkl"]
trained_model_name_notl = ["GTX1080_without_TL.pkl","RTX2070_without_TL.pkl","TitanX_without_TL.pkl",]
trained_model_name_driple = ["GTX1080_driple_without_TL.pkl","RTX2070_driple_without_TL.pkl","TitanX_driple_without_TL.pkl"]
#PATH = "./data/for_scheduling/" +" TitanX_without_TL.pkl" "TitanX_without_TL.pkl","RTX2070_without_TL.pkl","GTX1080_without_TL.pkl"
gnn_args = SimpleNamespace(**gnn_args)
#print(gnn_args)
model = GNN(**vars(gnn_args))
#print(model)
model.eval()
#preferred_GPU = []
t = tqdm(total=len(model_list), desc="input_graphs", leave=True, unit='predicted')
#(2, 2, 4, 8),
increment = [((10 - i) * 10) / 100 for i in range(11)]
'''vars_1 = [(GPU Mem Util, GPU Usage, CPU, Memory Usage)], values can go between 0 to 1 to add scaling factor for variables, 1 is normal, 0 means the parameter is excluded'''

vars_1 = [(1, 1, 1, 1)]

'''Evaluation'''
#vars_1 = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (0,1,1,1),(1,0,1,1),(1,1,0,1),(1,1,1,0),
#, (4,1,1,1),
#          (2, 2, 4, 4), (3, 3, 3, 9), (2, 2, 8, 8)]
for var_i in range(len(vars_1)):

    for j in range(len(model_list)):

        exec_time = []
        efficiency = []
        score = []
        total_exec_time = 0
        total_efficiency = []

        for inc_i in range(1):

            total_efficiency = []
            for trained_model in trained_model_name:
                
                PATH = "./data/for_scheduling/" + trained_model
                model.load_state_dict(torch.load(PATH))

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                #print("device:", device)
                model.cuda()
                move_cuda()
                output_val = 0.0
                output_temp = model(features['val'][j], adj['val'][j])
                output_temp = output_temp.tolist()

                t.update(1)


            #print(output_temp.tolist())
            #gpu mem util%, gpu util%,  cpu util% , gpu mem

                burst = []
                idle = []
                avg_peak = []
                e_time = []
                idx = 0
                efficiency_sum = 0
                for i in range(4):#2
                    burst.append({resource[i+0]:(value_category[0],output_temp[0][idx+0])})
                    idle.append({resource[i+0]:(value_category[1],output_temp[0][idx+1])})
                    avg_peak.append({resource[i+0]:(value_category[2],output_temp[0][idx+2])})
                    e_time.append({resource[i+0]:(value_category[3],output_temp[0][idx+3])})
                    total_exec_time += output_temp[0][idx+3]
                    if var_i > 9:
                        if vars_1[var_i][i] == 1:
                            efficiency_sum+=((output_temp[0][idx+2]/output_temp[0][idx+0]) * increment[inc_i])
                        else:
                            efficiency_sum+=((output_temp[0][idx + 2] / output_temp[0][idx + 0]) * ((1.0-increment[inc_i])/vars_1[var_i][i]))
                    else:
                        efficiency_sum+=((output_temp[0][idx + 2] / output_temp[0][idx + 0]) * vars_1[var_i][i])
                    idx+=4#3
                    if idx > 12:#6
                        break
                total_efficiency.append(efficiency_sum)
                exec_time.append(total_exec_time)
            print(total_efficiency)
            print(exec_time)
            """Efficiency"""
            max_val = max(total_efficiency)
            maxpos = total_efficiency.index(max_val)
            eff_GPU = trained_model_name_notl[maxpos].split('_')[0]
            output_eff = (eff_GPU,max_val)
            
        #loop start

#        for i in range(len(increment)):
 #           max_val = max([efficiency[0][i], efficiency[1][i], efficiency[2][i]])
  #          maxpos = [efficiency[0][i], efficiency[1][i], efficiency[2][i]].index(max_val)
  #          eff_GPU = trained_model_name_notl[maxpos].split('_')[0]
  #          output_eff = (eff_GPU,max_val)
  
            """Execution Time"""
            min_val = min(exec_time)
            minpos = exec_time.index(min_val)
            best_GPU = trained_model_name_notl[minpos].split('_')[0]
            output_best = (best_GPU,min_val)

        
        
            """preferred_GPU represents the GPU allocation based on the obtained scores"""
            preferred_GPU=({csv_header[0]:model_list[j],csv_header[1]:output_best[0],csv_header[2]:output_best[1]})     #exec time
            print(preferred_GPU)
            preferred_GPU = {csv_header[0]:model_list[j],csv_header[1]:output_eff[0],csv_header[3]:output_eff[1]}  #efficiency
            print(preferred_GPU)
        #preferred_GPU.append({csv_header[0]:model_list[j],csv_header[1]:output_best[0],csv_header[2]:output_best[1]})
    #t.close()
    
    """Evaluation"""
    
            # filename = "./data/efficiencytest/preferred_gpu_" + str(increment[inc_i]) +"_"+ '.'.join(map(str,vars_1[var_i])) + ".csv"
            # if check_if_file_exists(filename) == True:
                # with open(filename,'a+') as csv_file:
                    # writer = csv.DictWriter(csv_file, fieldnames=csv_header)
                    # writer.writerow(preferred_GPU)
            # else:
                # with open(filename,'a+') as csv_file:
                    # writer = csv.DictWriter(csv_file, fieldnames=csv_header)
                    # writer.writeheader()
                    # writer.writerow(preferred_GPU)
            # if var_i <10:
                # break
# #df_pref_gpu = pd.read_csv('.preferred_gpu.csv')
#df_pref_gpu = df_pref_gpu.sort_values(by=[csv_header[3]])
#df_pref_gpu.to_csv('.preferred_gpu_sorted.csv',index=False)
    # for i in range(len(increment)):
        # filename = "./data/efficiencytest/preferred_gpu_" + str(increment[i]) +"_"+ '.'.join(map(str,vars_1[var_i])) + ".csv"
        # get_JCT.calc_JCT(filename,increment[i],'.'.join(map(str,vars_1[var_i])))
        # df_pref_gpu = pd.read_csv(filename)
        # df_pref_gpu = df_pref_gpu.sort_values(by=[csv_header[3]])#3 efficiency, 2 execution time
        # df_pref_gpu.to_csv(filename+'_sorted.csv',index=False)
        # #get_JCT.calc_JCT(filename+'_sorted.csv', increment[i], '.'.join(map(str, vars_1[var_i])))
        # if var_i <10:
            # break
    # for i in range(len(increment)):
        # filename = "./data/efficiencytest/preferred_gpu_" + str(increment[i]) +"_"+ '.'.join(map(str,vars_1[var_i])) + ".csv"
        # get_JCT.calc_JCT(filename+'_sorted.csv', increment[i], '.'.join(map(str, vars_1[var_i])))
        # if var_i <10:
            # break
#loop end
