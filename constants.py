'''Hyperparameter Values'''
hp_model = ["vgg11","vgg16","vgg19","googlenet","_lenet","densenet40-k12","densenet100-k12","inception3","inception4","resnet20","resnet50","resnet101","dcgan","cvae","conditionalgan"] #for dissecting execution time
hp_model_node_freq = ["dcgan","cvae","conditionalgan","vgg11","vgg16","vgg19","googlenet","lenet","densenet40-k12","densenet100-k12","inception3","inception4","resnet20","resnet50","resnet101"] #for node operation evaluation
hp_model_training = ["lenet","densenet40-k12","densenet100-k12","inception3","inception4","resnet20","resnet50","resnet101"] #for training
hp_model_generative = ["dcgan","cvae","conditionalgan"] #for node operation evaluation
hp_model_scheduler = ["resnet32","gan","trivial","alexnet"]
#"vgg11","vgg16","vgg19","googlenet",
gpu_model = ["GTX1080","RTX2070","TitanX"]
gpu_model_training = ["GTX1080"] #Change values accordingly to which GPU is being used for training
hp_batch_size = [8,16,32,64]
hp_optimizer = ['sgd','adam']
hp_epoch = [10,20]
hp_dataset = ["mnist","cifar10","imagenet"]

''' File Configurations'''
csv_header = ['GPU Util%','GPU Mem Util%', 'GPU Memory','CPU Util%','tx','rx']

perf_file_header = ["GPU","Model","BatchSize","Optimizer","Epoch","Dataset","Resource","avg Idle time","avg Active time","avg Peak consumption","Execution Time"]
hyperparam_header = ['GPU', 'Model','Batch Size','Optimizer','Epoch','Dataset','Execution Time'] #execution time
hyperparam_header_2 = ['GPU', 'Model','Batch Size','Optimizer','Epoch','Dataset','Operation Count'] #op count
node_frequency_header = ['Operation','Frequency']


'''Paths and Filenames'''  #{0} for gpu model {1} for dataset i.e. GTX1080_cifar10
real_data_path = './data/Real Data/{0}_{1}/'
scheduler_data_path = './data/scheduler-data/data/{0}_{1}/'
driple_data_path = './data/driple_results/'
training_data_path = './data/{0}_{1}/'
graphs_data_path = './data/Real Data/{0}_{5}/graphs/{0}_{1}_{2}_{3}_{4}_{5}.pbtxt' # {0} gpu, {1} model, {2} bsize, {3} opt, {4} _epoch , {5} dataset
graphs_generator_data_path = './data/Real Data/{0}_{5}/graphs/{0}_{1}_{2}_{3}_{4}_{5}_generator.pbtxt' # {0} gpu, {1} model, {2} bsize, {3} opt, {4} _epoch , {5} dataset
graphs_discriminator_data_path = './data/Real Data/{0}_{5}/graphs/{0}_{1}_{2}_{3}_{4}_{5}_discriminator.pbtxt' # {0} gpu, {1} model, {2} bsize, {3} opt, {4} _epoch , {5} dataset
graphs_decoder_data_path = './data/Real Data/{0}_{5}/graphs/{0}_{1}_{2}_{3}_{4}_{5}_decoder.pbtxt' # {0} gpu, {1} model, {2} bsize, {3} opt, {4} _epoch , {5} dataset
graphs_encoder_data_path = './data/Real Data/{0}_{5}/graphs/{0}_{1}_{2}_{3}_{4}_{5}_encoder.pbtxt' # {0} gpu, {1} model, {2} bsize, {3} opt, {4} _epoch , {5} dataset
logfile_name = '_logfile.txt'
exec_time_filename = "exec_time_weighted.csv"
node_frequency_filename = '{0}_node-frequency.csv' # {0} gpu
operation_count_filename = 'operation_count.csv'
training_perf_filename = './data/training_perf_file_new.csv'
driple_dataset_path = './driple/training/'

'''Command-Line Arguments'''
cmd_gpu = "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv"

'''Text Entries'''
process_start_report = "Process has started\n"
job_start_report_driple = '{0}_{1} has Started at {2}\n' # {0} gpu, {1} model, {2} time
job_done_report_driple =  '{0}_{1} has Finished at {2}'  # {0} gpu, {1} model, {2} time
job_start_report = '{0}_{1}_{2}_{3}_{4}_{5} has Started at {6}\n' # {0} gpu, {1} model, {2} bsize, {3} opt, {4} _epoch , {5} dataset, {6} time
job_done_report =  '{0}_{1}_{2}_{3}_{4} has Finished at {5}\n'  # {0} gpu, {1} model, {2} bsize, {3} opt, {4} _epoch, {5} time
job_done_report_exec =  '{0}_{1}_{2}_{3}_{4} has Finished at '  # {0} gpu, {1} model, {2} bsize, {3} opt, {4} _epoch, {5} time
execution_time_entry = 'Execution Time: {0}\n' #{0} Execution Time
command_driple_train = "python ./driple/training/gcn.py --variable --gru --epochs=100000 --patience=1000 --variable_conv_layers=Nover2 --only_graph --hidden=64 --mlp_layers=3 --fixed "
