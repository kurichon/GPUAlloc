from models import model
from models import alexnet_model as alex
#import densenet_model
from models import googlenet_model
from models import lenet_model
from models import resnet_model
from models import trivial_model
from models import vgg_model

import tensorflow as tf
from tensorflow.python.tools import freeze_graph

def generate_frozen_graph(model,model_name):
    #model = TrainingModel.start_cnn()
#    model = TrainingModel.train_resnet50()
 
    tf.saved_model.save(model, "test")    
    
    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
    
    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    
    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)
    
    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)
    
    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./",
                      name=model_name+".pb",
                      as_text=False)
        
    

if __name__ == '__main__':
    model = alex.AlexnetModel()
    generate_frozen_graph(model,"alexnet")
    print (model)



