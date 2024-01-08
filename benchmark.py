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
#import networks.cvae as cvae
#import networks.conditional_gan as cond_gan
#import networks.dcgan as dcgan

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
#constants.hp_model_imagenet =["densenet40-k12","densenet100-k12","resnet20"]
#constants.hp_model_cifar = ["densenet40-k12","densenet100-k12","resnet20","resnet50","resnet101"]
#constants.hp_model = ["vgg11","vgg16","vgg19","googlenet","lenet","densenet40-k12","densenet100-k12","inception3","inception4","resnet20","resnet50","resnet101"]
#constants.hp_model_cifar = ["vgg11"]#,#"lenet"]
#constants.gpu_model_training = ["GTX1080"]#,"RTX2070","TitanX"]
#constants.hp_batch_size = [8,16,32,64]
#constants.hp_optimizer = ['sgd','adam']
#constants.hp_epoch = [10,20]
#constants.hp_dataset = ["cifar10","imagenet"]
hp_new_dataset = ["mnist"]
hp_new_models = ["gan"]


hp_test_dataset = ["imagenet","cifar10"]
hp_test_model = ["resnet32","trivial","alexnet"]
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
    
    with open(data_path + timestr + constants.logfile_name, 'w+') as logfile:
        logfile.write(constants.process_start_report)
    if test_flag == True:
            for _dset in hp_test_dataset:
                for test_model in tqdm(hp_test_model,desc='Test_Models'):
                    for bsize in constants.hp_batch_size:
                        for _epoch in constants.hp_epoch:
                            for opt in constants.hp_optimizer:
                            
                                start_time = datetime.datetime.now()
                                data_path = './data/{0}/'.format("eval_graphs")
                                
                                check_if_directory_exists(data_path)
                                        
                                with open(data_path + timestr + '_logfile.txt', 'a+') as logfile:
                                    logfile.writelines(
                                        constants.job_start_report.format(
                                        "eval", test_model, bsize, opt, _epoch, _dset, start_time))
                                
                                code_name = "{0}_{1}_{2}_{3}_{4}_{5}".format("eval", test_model, bsize, opt, _epoch, _dset)
                                newfile_flag = True
                                stop_threads = False                     
                                get_gpu_resource_every_second()
                                
                                # --gpu_memory_frac_for_testing 0.3 limits gpu usage
                                """Do stuff model bsize opt epoch"""
                                COMMAND = "python ./tf_cnn_benchmarks/tf_cnn_benchmarks.py --graph_file ./data/eval_graphs/{0}.pbtxt --data_format=NCHW --variable_update=replicated --nodistortions --gradient_repacking=8 --num_gpus=1 --weight_decay=1e-4 ".format(
                                code_name) + "--data_dir=${data_dir} --train_dir=${ckpt_dir}" + " --model={0} --batch_size={1} --optimizer={2} --num_epochs=0 --data_name={3}".format(
                                test_model, bsize, opt,_dset)
                                os.system(COMMAND)
                                """Do stuff"""
                                        
                                end_time = datetime.datetime.now()
                                        
                                with open(data_path + timestr + constants.logfile_name, 'a+') as logfile:
                                    logfile.writelines(
                                    constants.job_start_report.format(
                                    "eval", test_model, bsize, opt, _epoch, _dset, start_time))
                                    logfile.writelines(constants.execution_time_entry.format(end_time-start_time))
                                """Buffer time to show unloading of model"""
                                print("Wait for buffer time")
                                time.sleep(5)
                                """New model"""
                                stop_threads = True
                                time.sleep(30) #5 minutes
                                    
    else:
        for gpu in constants.gpu_model_training:
            for _dset in hp_new_dataset:
                #if _dset == "imagenet":
                #    models_to_load = constants.hp_model_imagenet
                #elif _dset == "cifar10":
                #    models_to_load = constants.hp_model_cifar
                for model in tqdm(hp_new_models,desc='models'):
                    for bsize in constants.hp_batch_size:
                        for _epoch in constants.hp_epoch:
                            for opt in constants.hp_optimizer:
                                
                                start_time = datetime.datetime.now()
                                data_path = constants.training_data_path.format(gpu,_dset)
                                
                                check_if_directory_exists(data_path)
                                
                                try:
                                    os.makedirs(data_path)
                                except FileExistsError:
                                    # directory already exists
                                    pass
                                
                                
                                with open(data_path + timestr + constants.logfile_name, 'a+') as logfile:
                                    logfile.writelines(
                                        constants.job_start_report.format(
                                        gpu, model, bsize, opt, _epoch, _dset, start_time))
                                
                                code_name = "{0}_{1}_{2}_{3}_{4}_{5}".format(gpu, model, bsize, opt, _epoch, _dset)
                                newfile_flag = True
                                stop_threads = False                     
                                get_gpu_resource_every_second()
                
                                """Do stuff model bsize opt epoch"""
                                if model == 'cvae':
                                    sub_command= "import networks.cvae as cvae; cvae.train_vae('{0}',{1},{2},'{3}')".format(str(gpu),
                                    _epoch,bsize,str(opt))
                                    print(sub_command)
                                    COMMAND = "python -c \"\"\"{0}\"\"\"".format(sub_command)
                                    print(COMMAND)
                                    #cvae.train_vae(gpu,_epoch,bsize,opt)
                                elif model == 'dcgan':
                                    
                                    sub_command= "import networks.dcgan as dcgan; dcgan.train_gan('{0}',{1},{2},'{3}')".format(str(gpu),
                                    _epoch,bsize,str(opt))
                                    COMMAND = "python -c \"\"\"{0}\"\"\"".format(sub_command)
                                    
                                    #COMMAND = "python -c 'import networks.dcgan as dcgan; dcgan.train_gan({0},{1},{2},{3})'".format(gpu,_epoch,bsize,opt)
                                    #dcgan.train_gan(gpu,_epoch,bsize,opt)
                                elif model == 'conditionalgan':
                                    
                                    sub_command= "import networks.conditional_gan as cond_gan; cond_gan.train_cond_gan('{0}',{1},{2},'{3}')".format(gpu,
                                    _epoch,bsize,opt)
                                    COMMAND = "python -c \"\"\"{0}\"\"\"".format(sub_command)
                                elif model == 'gan':
                                    
                                    sub_command= "import networks.dcgan as dcgan; dcgan.train_gan('{0}',{1},{2},'{3}',True)".format(str(gpu),
                                    _epoch,bsize,str(opt))
                                    COMMAND = "python -c \"\"\"{0}\"\"\"".format(sub_command)
                                    #COMMAND = "python -c 'import networks.conditional_gan as cond_gan; cond_gan.train_cond_gan({0},{1},{2},{3})'".format(gpu,_epoch,bsize,opt)                           
                                    #cond_gan.train_cond_gan(gpu,_epoch,bsize,opt)
                                    
                                elif model =='lstm':                                    
                                    sub_command= "import networks.anomalyd as lstm; lstm.train_lstm('{0}',{1},{2},'{3}',True)".format(str(gpu),
                                    _epoch,bsize,str(opt))
                                    COMMAND = "python -c \"\"\"{0}\"\"\"".format(sub_command)
                                

                                #COMMAND = "python ./tf_cnn_benchmarks/tf_cnn_benchmarks.py --graph_file ./data/{1}_{2}/graphs/{0}.pbtxt --data_format=NCHW --variable_update=replicated --nodistortions --gradient_repacking=8 --num_gpus=1 --weight_decay=1e-4 ".format(
                                #code_name,gpu,_dset) + "--data_dir=${data_dir} --train_dir=${ckpt_dir}" + " --model={0} --batch_size={1} --optimizer={2} --num_epochs={3} --data_name={4}".format(
                                #model, bsize, opt, _epoch,_dset)
                                os.system(COMMAND)
                                """Do stuff"""
                                #print(COMMAND)
                                end_time = datetime.datetime.now()
                                
                                with open(data_path + timestr + constants.logfile_name, 'a+') as logfile:
                                    logfile.writelines(
                                        constants.job_done_report.format(
                                        gpu, model, bsize, opt, _epoch, end_time))
                                    logfile.writelines(constants.execution_time_entry.format(end_time-start_time))
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
code_name = "{0}_{1}_{2}_{3}_{4}_{5}".format(constants.gpu_model_training[0], hp_new_models[0], constants.hp_batch_size[0], constants.hp_optimizer[0], constants.hp_epoch[0], hp_new_dataset[0])
data_path = constants.training_data_path.format(constants.gpu_model_training[0],hp_new_dataset[0])
get_gpu_resource_every_second()
jobs = len(constants.hp_model)*len(constants.hp_batch_size)*len(constants.hp_optimizer)*2

start_training(test_run)

stop_threads = True
