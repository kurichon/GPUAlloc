import subprocess as sp
import os
from threading import Thread , Timer
import sched, time
import csv
import pbfile 
#from tf_cnn_benchmarks import tf_cnn_benchmarks as tf_cnn
csv_header = ['GPU Util%', 'GPU Memory']
timestr = time.strftime("%Y%m%d-%H%M%S")
#time_count = 1
with open('./data/' + timestr + '.csv', 'w+') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=csv_header)
    writer.writeheader()
# def get_gpu_memory():
#     output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
#     ACCEPTABLE_AVAILABLE_MEMORY = 1024
#     COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
#     try:
#         memory_use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
#     except sp.CalledProcessError as e:
#         raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
#     memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]
#     # print(memory_use_values)
#     return str(memory_use_values[0])
#
# def get_gpu_utilization():
#     output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
#     ACCEPTABLE_AVAILABLE_MEMORY = 1024
#     COMMAND = "nvidia-smi --query-gpu=utilization.gpu --format=csv"
#     try:
#         gpu_util_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
#     except sp.CalledProcessError as e:
#         raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
#     gpu_util_values = [int(x.split()[0]) for i, x in enumerate(gpu_util_info)]
#     # print(memory_use_values)
#     return str(gpu_util_values[0])

def get_elapsed_time():
     get_elapsed_time.counter +=1
     return str(get_elapsed_time.counter)
get_elapsed_time.counter = -1

def get_gpu_resource():
    
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv"
    try:
        gpu_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    gpu_util_values = [int(x.split()[0]) for i, x in enumerate(gpu_info)]
    memory_use_values = [int(x.split()[2]) for i, x in enumerate(gpu_info)]

    return [str(gpu_util_values[0]),str(memory_use_values[0])]
def get_gpu_resource_every_second():
    """
        This function calls itself every 1 sec and print the gpu_memory and gpu_util.
    """
    gpu_resource = get_gpu_resource()
    #print(get_elapsed_time())
   # gpu_resource = [get_gpu_utilization(),get_gpu_memory()]
    print (gpu_resource)
    with open('./data/' + timestr + '.csv', 'a+') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_header)
        #writer = writer.writerow(gpu_resource)
        #writer = writer.writerow({'GPU Util%': get_gpu_utilization(), 'GPU Memory': get_gpu_memory()})
        writer = writer.writerow({'GPU Util%': gpu_resource[0], 'GPU Memory': gpu_resource[1]})

    global stop_threads
    if stop_threads:
        return
    else:
        Timer(1/6, get_gpu_resource_every_second).start()

        #writer = writer.writerow([get_gpu_utilization(),get_gpu_memory()])
#def run_thread():
#    while True:
#        print("GPU Thread Running")
#        global stop_threads
#        if stop_threads:
#            break
#csv_header = ['GPU Util%', 'GPU Memory']
#timestr = time.strftime("%Y%m%d-%H%M%S")
#with open('./data/' + timestr + '.csv', 'w+') as csv_file:
#    writer = csv.writer(csv_file)
#    writer = writer.writerow(csv_header)
    #writer = writer.writerow(get_gpu_resource_every_second())

    
"""
Do stuff.
"""
"""Hyperparameters"""
hp_batch_size = [8,16,32,64]
hp_optimizer = ['momentum','sgd','rmsprop','adam']
hp_epoch = 10
hp_fp16_flag = [True,False]
hp_model =["AlexNet,"]
#1 iteration
stop_threads = False
get_gpu_resource_every_second()
DNN = pbfile.DeepLearningNetworks()
model = DNN.start_cnn()
stop_threads = True
