import pandas as pd
import os
import matplotlib.pyplot as plt
import re
from datetime import datetime
import csv
import constants
from tqdm import tqdm

#start = datetime.strptime("2023-10-21 01:43:59.850471", '%Y-%m-%d %H:%M:%S.%f')
#end = datetime.strptime("2023-10-21 02:15:20.198397", '%Y-%m-%d %H:%M:%S.%f')
#print(end - start)
#exit()
#optimize constants make constant.py
#create representation per model

#exec_time_filename = "exec_time.csv"

#hyperparam_header = ['GPU', 'Model','Batch Size','Optimizer','Epoch','Dataset','Execution Time']

file_list = []
dfs = []
#os.listdir('dir_path'): Return the list of files and directories in a specified directory path.

#hp_model = ["vgg11","vgg16","vgg19","googlenet","_lenet","densenet40-k12","densenet100-k12","inception3","inception4","resnet20","resnet50","resnet101"]

#gpu_model = ["GTX1080","RTX2070","TitanX"]
#hp_batch_size = [8,16,32,64]
#hp_optimizer = ['sgd','adam']
#hp_epoch = [10,20]
#hp_dataset = ["cifar10","imagenet"]
exec_time = []
#filename =  "/data_sample.csv"
#path = 'data/Real Data/GTX1080_cifar10/'
#path = 'data/GTX1080_cifar10/'
#path = 'data/TitanX_cifar10/'
#path = 'data/GTX1080_cifar10/'
#path = 'data/GTX1080_cifar10/'
#list_of_logfiles = os.listdir(path)
list_of_logfiles = ''
for gpu in constants.gpu_model:
    for dataset in constants.hp_dataset:        
        path = constants.real_data_path.format(gpu,dataset) 
        for subdir, dirs, files in os.walk(path):
            for file in files:
                if constants.logfile_name in file:
                    list_of_logfiles += os.path.join(subdir,file) + '\n'
list_of_logfiles = list(filter(None,list_of_logfiles.split('\n')))
print(list_of_logfiles)


list_exec_time = []

for gpu in constants.gpu_model:
        for _dset in constants.hp_dataset:
            #if _dset == "imagenet":
            #    models_to_load = hp_model_imagenet
            #elif _dset == "cifar10":
            #    models_to_load = hp_model_cifar
            for model in tqdm(constants.hp_model,desc='models'):
                for bsize in constants.hp_batch_size:
                    for _epoch in constants.hp_epoch:
                        for opt in constants.hp_optimizer:
                            for logfile in list_of_logfiles:
                                with open(logfile,'r') as _file: 
                                    for line in _file:  
                                        
                                        if constants.job_done_report.format(
                                        gpu, model, bsize, opt, _epoch)in line:
                                            
                                            dataset = _file.name.split("_")[1].split("/")[0]
                                            exec_time = next(_file).split(": ")[1].replace("\n","")
                                            exec_time = sum(float(x) * 60 ** i for i, x in enumerate(reversed(exec_time.split(':'))))
                                            #print(secs)
                                            #exec_time = datetime.strptime(exec_time,'%H:%M:%S.%f')
                                            #exec_time = exec_time.timestamp()
                                            list_exec_time.append({constants.hyperparam_header[0]: gpu, constants.hyperparam_header[1]: model, constants.hyperparam_header[2]: bsize,
                                            constants.hyperparam_header[3]: opt, constants.hyperparam_header[4]: _epoch, constants.hyperparam_header[5]: dataset ,constants.hyperparam_header[6] : exec_time})

                                        #update to csv format
    
print(list_exec_time)
with open(constants.exec_time_filename, 'w+') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=constants.hyperparam_header)
    writer.writeheader()
    writer.writerows(list_exec_time)

