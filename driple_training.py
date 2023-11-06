import subprocess as sp
import os
from threading import Thread , Timer
import sched, time
import csv
#import pbfile
from tqdm import tqdm
import datetime
import psutil
import constants
import sys

#from tf_cnn_benchmarks import tf_cnn_benchmarks as tf_cnn
#constants.csv_header = ['GPU Util%','GPU Mem Util%', 'GPU Memory','CPU Util%','tx','rx']

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
    COMMAND = constants.cmd_gpu
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
            writer = csv.DictWriter(csv_file, fieldnames=constants.csv_header)
            writer.writeheader()
        newfile_flag = False
    gpu_resource = get_gpu_resource()
    
    #print (gpu_resource)
    with open(data_path + code_name + '.csv', 'a+') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=constants.csv_header)
        writer = writer.writerow({constants.csv_header[0]: gpu_resource[0], constants.csv_header[1]: gpu_resource[1],constants.csv_header[2]: gpu_resource[2],
        constants.csv_header[3]: gpu_resource[3],constants.csv_header[4]: 0, constants.csv_header[5]: 0})

    global stop_threads
    if stop_threads:
        return
    else:
        Timer(1/6, get_gpu_resource_every_second).start()



"""Hyperparameters"""
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
'''0: proposed,1: original method, 2: possible best combination', 3 all parameters'''
        
def start_driple_training():
    for iteration in range(2):
        for gpu in constants.gpu_model:
                            
            start_time = datetime.datetime.now()
            data_path = constants.driple_data_path
            check_if_directory_exists(data_path)
            try:
                os.makedirs(data_path)
            except FileExistsError:
                # directory already exists
                pass
            
            
            with open(data_path + timestr + constants.logfile_name, 'a+') as logfile:
                logfile.writelines(
                constants.job_start_report_driple.format(
                gpu, iteration, start_time))
            
                code_name = "driple_{0}_{1}".format(gpu, iteration)
                newfile_flag = True
                stop_threads = False                     
                get_gpu_resource_every_second()
                sys.stdout = open(data_path + code_name +constants.logfile_name,'w')
                """Do stuff model bsize opt epoch"""
                COMMAND = constants.command_driple_train + "--data={0}{1}_{2}.pkl".format(constants.driple_dataset_path,gpu,iteration)
                os.system(COMMAND)
                """Do stuff"""
                #print(COMMAND)
                end_time = datetime.datetime.now()
                sys.stdout.close()
                with open(data_path + timestr + constants.logfile_name, 'a+') as logfile:
                    logfile.writelines(
                        constants.job_done_report_driple.format(
                        gpu, iteration, end_time))
                    logfile.writelines(constants.execution_time_entry.format(end_time-start_time))
                """Buffer time to show unloading of model"""
                time.sleep(5)  
                """New model"""
                
                stop_threads = True
                time.sleep(30) #5 minutes
                        
                """ End 1 iteration"""


    
"""
Do stuff.
Start thread
"""


test_run = False
newfile_flag = True
stop_threads = False
code_name = "driple_{0}_{1}".format(constants.gpu_model[0], 0) #0 is proposed model
data_path = constants.driple_data_path
get_gpu_resource_every_second()

start_driple_training()

stop_threads = True
