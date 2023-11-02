#create pb file
import os
import numpy as np
import networkx as nx
import tensorflow as tf
import matplotlib.pyplot as plt
from driple.dataset.dataset_builder import parsing as parse
import constants
from tqdm import tqdm
import csv
import itertools
def nx_node_using_op_node_name(op_list):
    nx_node_form = []
    for op in op_list:
        nx_node_form.append((op, {"tensor_size":0, "node_type":0}))
    return nx_node_form


def nx_edge_using_op_node_edge(edge_list):
    return edge_list

if __name__ == '__main__':
    #pbtxt_data = parse.import_pbtxt('./data/data/GTX1080_cifar10/graphs/GTX1080_densenet40-k12_8_adam_10_cifar10.pbtxt')
    #print(op_node_name)
    #op_node_name = parse.pbtxt_extract_op_node_name(pbtxt_data)
    #op_node_edge = parse.pbtxt_extract_op_node_edge(pbtxt_data)
    
    #op_name = parse.pbtxt_extract_op_name(pbtxt_data)
    #unique_op = list(set(op_name))
    #node_frequency = [op_name.count(operand)/len(op_name)for operand in unique_op]
    
    dataset_op_count_list = []
    for gpu in constants.gpu_model:
        dataset_op_name_list = []
        node_frequency_list = []
        for _dset in constants.hp_dataset:
            #if _dset == "imagenet":
        #    models_to_load = hp_model_imagenet
        #elif _dset == "cifar10":
        #    models_to_load = hp_model_cifar
            for model in tqdm(constants.hp_model_node_freq,desc='models'):
                for bsize in constants.hp_batch_size:
                    for _epoch in constants.hp_epoch:
                        for opt in constants.hp_optimizer:
                            #print(constants.graphs_data_path.format(gpu,model,bsize,opt,_epoch,_dset))
                            try:
                                pbtxt_data = parse.import_pbtxt(constants.graphs_data_path.format(gpu,model,bsize,opt,_epoch,_dset))
                                op_name = parse.pbtxt_extract_op_name(pbtxt_data)
                                dataset_op_name_list.append(op_name)
                                dataset_op_count_list.append({constants.hyperparam_header_2[0]: gpu, constants.hyperparam_header_2[1]: model, constants.hyperparam_header_2[2]: bsize,
                                        constants.hyperparam_header_2[3]: opt, constants.hyperparam_header_2[4]: _epoch, constants.hyperparam_header_2[5]: _dset , constants.hyperparam_header_2[6]: len(op_name)})
     #   print(dataset_op_name_list[0])
                            except FileNotFoundError:
                                pass 
        
        op_occurence = list(itertools.chain.from_iterable(dataset_op_name_list))
        unique_op = list(set(op_occurence))
        
        node_frequency = [op_occurence.count(operand)/len(op_occurence)for operand in unique_op]
        node_frequency_result = np.column_stack((unique_op,node_frequency))
        #print(node_frequency_result)
        for i in range(len(node_frequency_result)):
            node_frequency_list.append({constants.node_frequency_header[0]:node_frequency_result[i][0],
            constants.node_frequency_header[1]:node_frequency_result[i][1]})
        with open(constants.node_frequency_filename.format(gpu), 'w+') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=constants.node_frequency_header)
            writer.writeheader()
            writer.writerows(node_frequency_list)
    #Write all op count
    with open(constants.operation_count_filename, 'w+') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=constants.hyperparam_header_2)
        writer.writeheader()
        writer.writerows(dataset_op_count_list)
        
        
    
    
    
    
   # arr = np.column_stack((unique_op,node_frequency))
   # print([unique_op,node_frequency])
   # print(arr)
    #print(op_node_edge)
    #nx_node = nx_node_using_op_node_name(op_node_name)
    #nx_edge = nx_edge_using_op_node_edge(op_node_edge)

    #g = nx.MultiGraph()
    #g.add_nodes_from(nx_node)
    #g.add_edges_from(nx_edge)
    #g = nx.complete_graph(5)
    #nx.draw(g)
    #plt.show()
   
    
