import subprocess as sp
import os
from threading import Thread , Timer
import sched, time
import csv
#import pbfile
from tqdm import tqdm
import datetime
import psutil
#from tf_cnn_benchmarks import tf_cnn_benchmarks as tf_cnn
csv_header = ['GPU Util%','GPU Mem Util%', 'GPU Memory','CPU Util%','tx','rx']
timestr = time.strftime("%Y%m%d-%H%M%S")

def check_if_directory_exists(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        # directory already exists
        pass    
        
def get_elapsed_time():
     get_elapsed_time.counter +=1
     return str(get_elapsed_time.counter)
get_elapsed_time.counter = -1

def get_gpu_resource():
    
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv"
    try:
        gpu_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    gpu_util_values = [int(x.split()[0]) for i, x in enumerate(gpu_info)]
    memory_util_values = [int(x.split()[2]) for i, x in enumerate(gpu_info)]
    memory_use_values = [int(x.split()[4]) for i, x in enumerate(gpu_info)]

    return [str(gpu_util_values[0]),str(memory_util_values[0]),str(memory_use_values[0]),str(psutil.cpu_percent(percpu=False, interval=1/6))]
def get_gpu_resource_every_second():
    """
        This function calls itself every 1 sec and print the gpu_memory and gpu_util.
    """
    global code_name
    global data_path
    global newfile_flag
    
    
    check_if_directory_exists(data_path)
    
    
    if newfile_flag == True:
        with open(data_path + code_name + '.csv', 'w+') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_header)
            writer.writeheader()
        newfile_flag = False
    gpu_resource = get_gpu_resource()
    
    #print (gpu_resource)
    with open(data_path + code_name + '.csv', 'a+') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_header)
        writer = writer.writerow({'GPU Util%': gpu_resource[0], 'GPU Mem Util%': gpu_resource[1],'GPU Memory': gpu_resource[2],'CPU Util%': gpu_resource[3],'tx': 0, 'rx': 0})

    global stop_threads
    if stop_threads:
        return
    else:
        Timer(1/6, get_gpu_resource_every_second).start()



"""Hyperparameters"""
hp_model_imagenet =["vgg11","vgg16","vgg19","googlenet","lenet","alexnet","trivial"]
hp_model_cifar = ["alexnet","trivial","resnet20_v2","resnet20","resnet32","resnet44","resnet56","resnet110"]
hp_model = ["vgg11","vgg16","vgg19","googlenet","lenet","densenet40-k12","densenet100-k12","inception3","inception4","resnet20","resnet50","resnet101"]
#hp_model_cifar = ["vgg11"]#,#"lenet"]
gpu_model = ["GTX1080"]#,"RTX2070","TitanX"]
hp_batch_size = [8,16,32,64]
hp_optimizer = ['sgd','adam']
hp_epoch = [10,20]
hp_dataset = ["cifar10","imagenet"]
hp_test_dataset = ["imagenet"]
hp_test_model = ["resnet20"]
"""
Add --use-fp16
"""
#_hp_fp16_flag = [True,False]

"""Other parameters"""
hp_data_format = "NCHW"
hp_variable_update = "replicated"
hp_decay_rate = "1e-4"
hp_num_gpus = 1
hp_data_dir = "${DATA_DIR}"
hp_train_dir = "${CKPT_DIR}"
"""-nodistortions"""

def start_training(test_flag):
    global stop_threads
    global data_path
    global newfile_flag
    global code_name
    check_if_directory_exists(data_path)
    
    with open(data_path + timestr + '_logfile.txt', 'w+') as logfile:
        logfile.write("Process has started\n")
    if test_flag == True:
        for gpu in gpu_model:
            for _dset in hp_test_dataset:
                    for test_model in tqdm(hp_test_model,desc='Test_Models'):
                                
                        start_time = datetime.datetime.now()
                        data_path = './test_data/{0}_{1}/'.format(gpu,_dset)
                        
                        check_if_directory_exists(data_path)
                                
                        with open(data_path + timestr + '_logfile.txt', 'a+') as logfile:
                            logfile.writelines(
                                "{0}_{1}_{2}_{3}_{4}_{5} has Started at {6}\n".format(
                                'TestGPU', test_model, '32', 'sgd', '1', _dset, start_time))
                        
                        code_name = "{0}_{1}_{2}_{3}_{4}_{5}".format('TestGPU2', test_model, '32', 'sgd', '1', _dset)
                        newfile_flag = True
                        stop_threads = False                     
                        get_gpu_resource_every_second()
                        
                        # --gpu_memory_frac_for_testing 0.3 limits gpu usage
                        """Do stuff model bsize opt epoch"""
                        COMMAND = "python ./tf_cnn_benchmarks/tf_cnn_benchmarks.py --graph_file ./data/graph_models/{0}.pbtxt --data_format=NCHW --variable_update=replicated --nodistortions --gradient_repacking=8 --num_gpus=1 --weight_decay=1e-4 ".format(
                                code_name) + "--data_dir=${DATA_DIR} --train_dir=${CKPT_DIR}" + " --model={0} --batch_size=32 --optimizer={2} --num_epochs=5 --data_name={4}".format(
                        test_model, '32', 'sgd', '5',_dset)
                        os.system(COMMAND)
                        """Do stuff"""
                                
                        end_time = datetime.datetime.now()
                                
                        with open(data_path + timestr + '_logfile.txt', 'a+') as logfile:
                            logfile.writelines(
                                "{0}_{1}_{2}_{3}_{4} has Finished at {5}\n".format(
                                'TestGPU', test_model, '32', 'sgd', '1', end_time))
                            logfile.writelines("Execution Time: {0}\n".format(end_time-start_time))
                        """Buffer time to show unloading of model"""
                        print("Wait for buffer time")
                        time.sleep(5)
                        """New model"""
                        stop_threads = True
                        time.sleep(300) #5 minutes
                                    
    else:
        for gpu in gpu_model:
            for _dset in hp_dataset:
                #if _dset == "imagenet":
                #    models_to_load = hp_model_imagenet
                #elif _dset == "cifar10":
                #    models_to_load = hp_model_cifar
                for model in tqdm(hp_model,desc='models'):
                    for bsize in hp_batch_size:
                        for _epoch in hp_epoch:
                            for opt in hp_optimizer:
                                
                                start_time = datetime.datetime.now()
                                data_path = './data/{0}_{1}/'.format(gpu,_dset)
                                
                                check_if_directory_exists(data_path)
                                
                                try:
                                    os.makedirs(data_path)
                                except FileExistsError:
                                    # directory already exists
                                    pass
                                
                                
                                with open(data_path + timestr + '_logfile.txt', 'a+') as logfile:
                                    logfile.writelines(
                                        "{0}_{1}_{2}_{3}_{4}_{5} has Started at {6}\n".format(
                                        gpu, model, bsize, opt, _epoch, _dset, start_time))
                                
                                code_name = "{0}_{1}_{2}_{3}_{4}_{5}".format(gpu, model, bsize, opt, _epoch, _dset)
                                newfile_flag = True
                                stop_threads = False                     
                                get_gpu_resource_every_second()
                
                                """Do stuff model bsize opt epoch"""
                                COMMAND = "python ./tf_cnn_benchmarks/tf_cnn_benchmarks.py --graph_file ./data/{1}_{2}/graphs/{0}.pbtxt --data_format=NCHW --variable_update=replicated --nodistortions --gradient_repacking=8 --num_gpus=1 --weight_decay=1e-4 ".format(
                                code_name,gpu,_dset) + "--data_dir=${data_dir} --train_dir=${ckpt_dir}" + " --model={0} --batch_size={1} --optimizer={2} --num_epochs={3} --data_name={4}".format(
                                model, bsize, opt, _epoch,_dset)
                                os.system(COMMAND)
                                """Do stuff"""
                                print(COMMAND)
                                end_time = datetime.datetime.now()
                                
                                with open(data_path + timestr + '_logfile.txt', 'a+') as logfile:
                                    logfile.writelines(
                                        "{0}_{1}_{2}_{3}_{4} has Finished at {5}\n".format(
                                        gpu, model, bsize, opt, _epoch, end_time))
                                    logfile.writelines("Execution Time: {0}\n".format(end_time-start_time))
                                """Buffer time to show unloading of model"""
                                time.sleep(5)  
                                """New model"""
                                
                                stop_threads = True
                                time.sleep(5) #5 minutes
                                
                                """ End 1 iteration"""


    
"""
Do stuff.
Start thread
"""


test_run = False
newfile_flag = True
stop_threads = False
code_name = "{0}_{1}_{2}_{3}_{4}_{5}".format(gpu_model[0], hp_model_imagenet[0], hp_batch_size[0], hp_optimizer[0], hp_epoch[0], hp_dataset[0])
data_path = "./data/{0}_{1}/".format(gpu_model[0],hp_dataset[0])
get_gpu_resource_every_second()
iteration = (len(hp_model_imagenet)+len(hp_model_cifar))*len(hp_batch_size)*len(hp_optimizer)*2

start_training(test_run)

stop_threads = True
