import os
import torch
import random
import itertools
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import build_graph
import matplotlib.pyplot as plt

def find_model_list(pbtxt_path):
    pbtxt_files = []
    pbtxt_files.append([file_name[:-6] for file_name in os.listdir(pbtxt_path)])
    model_list = list(itertools.chain(*pbtxt_files))

    return model_list


def find_graph_label(result, split_name):
    GPU_Condition = (result["GPU"] == split_name[0])
    Model_Condition = (result["Model"] == split_name[1])
    Data_Set_Condition = (result["BatchSize"] == int(split_name[2]))
    Sync_Async_Condition = (result["Optimizer"] == split_name[3])
    Hyper_Parameter_Condition = (result["Epoch"] == int(split_name[4]))
    Job_Name_Condition = (result["Dataset"] == split_name[5])

    GPU_MEM_util_Condition = (result["Resource"] == "GPU Mem Util%")
    GPU_util_Condition = (result["Resource"] == "GPU Util%")
    Network_rx_Condition = (result["Resource"] == "CPU Util%")
    Network_tx_Condition = (result["Resource"] == "GPU Memory")

    GPU_MEM_util = result[GPU_Condition & Model_Condition & Data_Set_Condition & Sync_Async_Condition
                          & Hyper_Parameter_Condition & Job_Name_Condition & GPU_MEM_util_Condition]
    GPU_util = result[GPU_Condition & Model_Condition & Data_Set_Condition & Sync_Async_Condition
                      & Hyper_Parameter_Condition & Job_Name_Condition & GPU_util_Condition]
    Network_rx = result[GPU_Condition & Model_Condition & Data_Set_Condition & Sync_Async_Condition
                       & Hyper_Parameter_Condition & Job_Name_Condition & Network_rx_Condition]
    Network_tx = result[GPU_Condition & Model_Condition & Data_Set_Condition & Sync_Async_Condition
                        & Hyper_Parameter_Condition & Job_Name_Condition & Network_tx_Condition]

    return GPU_MEM_util, GPU_util, Network_rx, Network_tx


def in_result_file(result, before_model_list):
    after_model_list = []
    model_graph_label_dict = {}
    for model_name in before_model_list:
        split_name = model_name.split("_")
        GPU_MEM_util, GPU_util, Network_rx, Network_tx = find_graph_label(result, split_name)
        if len(GPU_MEM_util) + len(GPU_util) + len(Network_rx) + len(Network_tx) == 4:
            after_model_list.append(model_name)
            model_graph_label_dict[model_name] = [GPU_MEM_util["avg Idle time"],GPU_MEM_util["avg Active time"],
                                              GPU_MEM_util["avg Peak consumption"],GPU_MEM_util["Execution Time"],
                                              GPU_util["avg Idle time"],GPU_util["avg Active time"], GPU_util["avg Peak consumption"],GPU_util["Execution Time"],
                                              Network_rx["avg Idle time"], Network_rx["avg Active time"],
                                              Network_rx["avg Peak consumption"], Network_rx["Execution Time"],Network_tx["avg Idle time"],
                                              Network_tx["avg Active time"], Network_tx["avg Peak consumption"],Network_tx["Execution Time"]]

    return after_model_list, model_graph_label_dict


def save_graph(perf_result, b_size, n_of_groups, n_of_graphs, save_path, dataset_name):
    result = pd.read_csv(perf_result)

    pbtxt_folder_path = "features/"
    nodetype_file = "features/node-frequency.csv"
    model_list = find_model_list(pbtxt_folder_path)
    model_list, model_graph_label_dict = in_result_file(result, model_list)

    random.seed(42)
    random.shuffle(model_list)
    model_list = model_list[:n_of_graphs]

    train, val, test = np.split(model_list, [int(.6 * len(model_list)), int(.8 * len(model_list))])
    data_dict = {'train': train, 'val': val, 'test': test}
    #print(len(train))
    batch_size = b_size #default 32
    #batch_size= 32
    train_batch = int(len(train) / batch_size)
    val_batch = int(len(val) / batch_size)
    test_batch = int(len(test) / batch_size)
    n_graphs = {'train': [batch_size] * train_batch, 'val': [batch_size] * val_batch, 'test': [batch_size] * test_batch}

    adj = {}
    features = {}
    graph_labels = {}

    for dset in n_graphs:
        set_adj = [[] for _ in n_graphs[dset]]
        set_features = [[] for _ in n_graphs[dset]]
        set_graph_labels = [[] for _ in n_graphs[dset]]

        t = tqdm(total=np.sum(n_graphs[dset]), desc=dset, leave=True, unit=' graphs')
        for batch, batch_size in enumerate(n_graphs[dset]):

            for i in range(batch_size):
                model_name = data_dict[dset][i + batch * batch_size]
                g = build_graph.build_nx_graph(pbtxt_folder_path, nodetype_file, model_name)
                g.remove_nodes_from(list(nx.isolates(g)))
                #nx.draw(g)
                #plt.show()
                if nx.number_connected_components(g) > 1:
                    g = g.subgraph(max(nx.connected_components(g), key=len)).copy()

                dnn_name = model_name.split('_')[1]
                if dnn_name == "cvae" or dnn_name == "gan" or dnn_name == "dcgan" or dnn_name == "conditionalgan":
                    grouped_g = build_graph.group_nx_graph(g, 18)
                else:
                    grouped_g = build_graph.group_nx_graph(g, n_of_groups)
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

                graph_labels_temp = np.asarray(model_graph_label_dict[model_name]).flatten()
                set_adj[batch].append(adj_temp)
                set_features[batch].append(features_temp)
                set_graph_labels[batch].append(graph_labels_temp)

        t.close()

        adj[dset] = [torch.from_numpy(np.asarray(adjs)).float() for adjs in set_adj]
        features[dset] = [torch.from_numpy(np.asarray(fs)).float() for fs in set_features]
        graph_labels[dset] = [torch.from_numpy(np.asarray(gls)).float() for gls in set_graph_labels]

    print(features['train'])
    print(len(features['train']))
    print(len(features['train'][0]))
    print(len(features['train'][0][0]))
    print(len(features['train'][0][0][0]))

    save = os.getcwd()+ save_path + "/" + dataset_name
    with open(save, 'wb') as f:
        torch.save((adj, features, graph_labels), f)
        print(f)
