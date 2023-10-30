'''Hyperparameter Values'''
hp_model = ["vgg11","vgg16","vgg19","googlenet","_lenet","densenet40-k12","densenet100-k12","inception3","inception4","resnet20","resnet50","resnet101"] #for dissecting execution time
hp_model_node_freq = ["vgg11","vgg16","vgg19","googlenet","lenet","densenet40-k12","densenet100-k12","inception3","inception4","resnet20","resnet50","resnet101"] #for node operation evaluation
hp_model_training = ["vgg11","vgg16","vgg19","googlenet","lenet","densenet40-k12","densenet100-k12","inception3","inception4","resnet20","resnet50","resnet101"] #for training

gpu_model = ["GTX1080","RTX2070","TitanX"]
gpu_model_training = ["RTX2070","TitanX"] #Change values accordingly to which GPU is being used for training
hp_batch_size = [8,16,32,64]
hp_optimizer = ['sgd','adam']
hp_epoch = [10,20]
hp_dataset = ["cifar10","imagenet"]

''' File Configurations'''
csv_header = ['GPU Util%','GPU Mem Util%', 'GPU Memory','CPU Util%','tx','rx']
hyperparam_header = ['GPU', 'Model','Batch Size','Optimizer','Epoch','Dataset','Execution Time'] #execution time
hyperparam_header_2 = ['GPU', 'Model','Batch Size','Optimizer','Epoch','Dataset','Operation Count'] #op count
node_frequency_header = ['Operation','Frequency']


'''Paths and Filenames'''  #{0} for gpu model {1} for dataset i.e. GTX1080_cifar10
real_data_path = './data/Real Data/{0}_{1}/'
training_data_path = './data/{0}_{1}/'
graphs_data_path = './data/Real Data/{0}_{5}/graphs/{0}_{1}_{2}_{3}_{4}_{5}.pbtxt' # {0} gpu, {1} model, {2} bsize, {3} opt, {4} _epoch , {5} dataset
logfile_name = '_logfile.txt'
exec_time_filename = "exec_time.csv"
node_frequency_filename = '{0}_node-frequency.csv' # {0} gpu
operation_count_filename = 'operation_count.csv'


'''Command-Line Arguments'''
cmd_gpu = "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv"

'''Text Entries'''
process_start_report = "Process has started\n"
job_start_report = '{0}_{1}_{2}_{3}_{4}_{5} has Started at {6}\n' # {0} gpu, {1} model, {2} bsize, {3} opt, {4} _epoch , {5} dataset, {6} time
job_done_report =  '{0}_{1}_{2}_{3}_{4} has Finished at {5}\n'  # {0} gpu, {1} model, {2} bsize, {3} opt, {4} _epoch, {5} time
execution_time_entry = 'Execution Time: {0}\n' #{0} Execution Time
