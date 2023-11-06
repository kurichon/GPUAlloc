import pandas
import os
import pandas as pd
import matplotlib.pyplot as plt
import time
import generate_perf_file
import constants
from tqdm import tqdm
import csv
import itertools
import sys


# try:
    # csv_file = open(constants.training_perf_filename, 'r')
# except FileNotFoundError:
    # with open(constants.training_perf_filename,'w+') as csv_file:  
        # writer = csv.DictWriter(csv_file, fieldnames=csv_header)
        # writer.writeheader()
# finally:
    # csv_file.close()

model_iterations = len(constants.hp_batch_size) * len(constants.hp_optimizer) * len(constants.hp_epoch)
#print(model_iterations)
timestr = time.strftime("%Y%m%d-%H%M%S")
perf_list = []
df_exec = pd.read_csv(constants.exec_time_filename)

with open('./data/' + timestr + '_average_logfile.txt', 'w+') as logfile:
    logfile.write("Process has started\n")
    
    for gpu in constants.gpu_model:
        for _dset in constants.hp_dataset:
            for model in tqdm(constants.hp_model_node_freq,desc='models'):
                #print("model")
                average_mem = 0
                average_util = 0
                average_mem_util = 0
                average_cpu_util = 0
                for bsize in constants.hp_batch_size:
                    for opt in constants.hp_optimizer:
                        for _epoch in constants.hp_epoch:
                            code_name = "{0}_{1}_{2}_{3}_{4}_{5}".format(gpu, model, bsize, opt, _epoch, _dset)
                            gpu_dataset_path = constants.real_data_path.format(gpu,_dset)
                            """Read File"""
                            try:
                                #get execution time
                                exec_time = float((df_exec[(df_exec[constants.hyperparam_header[0]] == gpu) & (df_exec[constants.hyperparam_header[1]] == model) & (df_exec[constants.hyperparam_header[2]] == bsize)
                                & (df_exec[constants.hyperparam_header[3]] == opt) & (df_exec[constants.hyperparam_header[4]] == _epoch) &(df_exec[constants.hyperparam_header[5]] == _dset)][constants.hyperparam_header[6]]))
                                print(exec_time)
                                df = pd.read_csv(gpu_dataset_path + code_name+".csv")
                                df_new = df[["GPU Util%","GPU Mem Util%","GPU Memory","CPU Util%"]]
                                size = len(df_new)
                                average_util += df["GPU Util%"].sum()/size
                                average_mem += df["GPU Memory"].sum()/size
                                average_mem_util += df["GPU Mem Util%"].sum()/size
                                average_cpu_util += df["CPU Util%"].sum()/size
                                perf_list.append(generate_perf_file.generate_parameters(df_new, gpu,code_name,exec_time))
                                #logfile.writelines(code_name + "; Average GPU Util%, Gpu Memory= {0} , {1}\n".format(average_util, average_mem))
                            except FileNotFoundError:                
                                logfile.writelines(
                                    "{0}_{1}_{2}_{3}_{4}_{5}.csv does not exist\n".format(gpu, model, bsize, opt, _epoch,_dset))
                model_average_util = average_util/model_iterations 
                model_average_mem = average_mem/model_iterations 
                model_average_mem_util = average_mem_util/model_iterations
                model_average_cpu_util = average_cpu_util/model_iterations
                logfile.writelines("{0}_{1}_{2}; Average GPU Util%:{3} ,Average Mem Util%:{4} , Average Memory:{5}, Average CPU Util%:{6}\n"
                .format(gpu,model,_dset,model_average_util, model_average_mem_util,model_average_mem,model_average_cpu_util))
    print(perf_list)
    merged = list(itertools.chain(*perf_list))
    print(merged)
    try:            
        with open(constants.training_perf_filename,'w+') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=constants.perf_file_header)
            writer.writeheader()
            writer.writerows(merged)
    except FileNotFoundError:
        print("File was not found, please check the correct directory")
