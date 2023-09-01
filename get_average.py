import pandas
import os
import pandas as pd
import matplotlib.pyplot as plt
import time
import generate_perf_file
from tqdm import tqdm


headers = ['GPU Util%','GPU Memory']

hp_model_imagenet =["vgg11","vgg16","vgg19","lenet","googlenet","alexnet","trivial"]
hp_model_cifar = ["alexnet","trivial","resnet20_v2","resnet20","resnet32","resnet44","resnet56","resnet110"]
gpu_model = ["GTX1080","RTX2070","TitanX"]
hp_batch_size = [8,16,32,64]
hp_optimizer = ['sgd','adam']
hp_epoch = [10,20]
hp_dataset = ["imagenet","cifar10"]

BASE_PATH = os.getcwd() + "/data"
model_iterations = len(hp_batch_size) * len(hp_optimizer) * len(hp_epoch)
print(model_iterations)
timestr = time.strftime("%Y%m%d-%H%M%S")
with open('./data/' + timestr + '_average_logfile.txt', 'w+') as logfile:
    logfile.write("Process has started\n")
    
    for gpu in gpu_model:
        for _dset in hp_dataset:
            if _dset == "imagenet":
                for model in tqdm(hp_model_imagenet,desc='Imagenet_Models'):
                    #print("model")
                    average_mem = 0
                    average_util = 0
                    for bsize in hp_batch_size:
                        #print("batch")
                        for opt in hp_optimizer:
                            #print("optimizer")
                            for _epoch in hp_epoch:
                                code_name = "{0}_{1}_{2}_{3}_{4}_{5}".format(gpu, model, bsize, opt, _epoch, _dset)
                                gpu_dataset_path = "/{0}_{1}/".format(gpu,_dset)
                                """Read File"""
                                try: 
                                    df = pd.read_csv(BASE_PATH + gpu_dataset_path + code_name+".csv")
                                    df_new = df[["GPU Util%","GPU Memory"]]
                                    size = len(df_new)
                                    average_util += df["GPU Util%"].sum()/size
                                    average_mem += df["GPU Memory"].sum()/size
                                    generate_perf_file.generate_parameters(df_new, gpu,code_name)
                                    #logfile.writelines(code_name + "; Average GPU Util%, Gpu Memory= {0} , {1}\n".format(average_util, average_mem))
                                except FileNotFoundError:                
                                    logfile.writelines(
                                        "{0}_{1}_{2}_{3}_{4}_{5}.csv does not exist\n".format(gpu, model, bsize, opt, _epoch,_dset))
                    model_average_util = average_util/model_iterations 
                    model_average_mem = average_mem/model_iterations 
                    logfile.writelines("{0}_{1}_{2}; Average GPU Util%, Gpu Memory= {3} , {4}\n".format(gpu,model,_dset,model_average_util, model_average_mem))
                                
            if _dset == "cifar10":
                for model in tqdm(hp_model_cifar,desc='Cifar10_Models'):
                    #print("model")
                    average_mem = 0
                    average_util = 0
                    for bsize in hp_batch_size:
                        #print("batch")
                        for opt in hp_optimizer:
                            #print("optimizer")
                            for _epoch in hp_epoch:
                                code_name = "{0}_{1}_{2}_{3}_{4}_{5}".format(gpu, model, bsize, opt, _epoch, _dset)
                                gpu_dataset_path = "/{0}_{1}/".format(gpu,_dset)
                                """Read File"""
                                try: 
                                    df = pd.read_csv(BASE_PATH + gpu_dataset_path + code_name+".csv")
                                    df_new = df[["GPU Util%","GPU Memory"]]
                                    size = len(df_new)
                                    average_util += df["GPU Util%"].sum()/size
                                    average_mem += df["GPU Memory"].sum()/size
                                    generate_perf_file.generate_parameters(df_new, gpu,code_name+'.csv')
                                    #logfile.writelines(code_name + "; Average GPU Util%, Gpu Memory= {0} , {1}\n".format(average_util, average_mem))
                                except FileNotFoundError:                
                                    logfile.writelines(
                                        "{0}_{1}_{2}_{3}_{4}_{5}.csv does not exist\n".format(gpu, model, bsize, opt, _epoch,_dset))
                    model_average_util = average_util/model_iterations 
                    model_average_mem = average_mem/model_iterations 
                    logfile.writelines("{0}_{1}_{2}; Average GPU Util%, Gpu Memory= {3} , {4}\n".format(gpu,model,_dset,model_average_util, model_average_mem))
                    
