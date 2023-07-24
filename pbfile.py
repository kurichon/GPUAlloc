#create pb file
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.keras import datasets, layers, models

#### Necessary Imports for Neural Net 

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D,\
     Flatten, BatchNormalization, AveragePooling2D, Dense, Activation, Add 
from tensorflow.keras.models import Model
from tensorflow.keras import activations
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

from sklearn.model_selection import train_test_split 

class DeepLearningNetworks:
        
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    class_types = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck'] # from cifar-10 website
    #lrdecay = 1e-1             
    
    #show the plot of accuracy
    def plot_accuracy(self,model):
        f,ax=plt.subplots(2,1,figsize=(10,10)) 

        #Assigning the first subplot to graph training loss and validation loss
        ax[0].plot(model.history.history['loss'],color='b',label='Training Loss')
        ax[0].plot(model.history.history['val_loss'],color='r',label='Validation Loss')
        #Plotting the training accuracy and validation accuracy
        ax[1].plot(model.history.history['accuracy'],color='b',label='Training  Accuracy')
        ax[1].plot(model.history.history['val_accuracy'],color='r',label='Validation Accuracy')
        
        plt.legend()
        plt.show()
        
    def start_cnn(self):
            
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.summary()
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10))
        model.summary()
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        history = model.fit(self.train_images, self.train_labels, epochs=10, 
                            validation_data=(self.test_images, self.test_labels))
        #self.plot_accuracy(model)
        return model
       
    def train_alexnet(self):
            train_ds=tf.data.Dataset.from_tensor_slices((train_images,train_labels))
            test_ds=tf.data.Dataset.from_tensor_slices((test_images,test_labels))
    
    
    def res_identity(self,x, filters): 
        ''' renet block where dimension doesnot change.
        The skip connection is just simple identity conncection
        we will have 3 blocks and then input will be added
        '''
        x_skip = x # this will be used for addition with the residual block 
        f1, f2 = filters
        
        #first block 
        x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)
        
        #second block # bottleneck (but size kept same with padding)
        x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)
        
        # third block activation used after adding the input
        x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        # x = Activation(activations.relu)(x)
        
        # add the input 
        x = Add()([x, x_skip])
        x = Activation(activations.relu)(x)
        
        return x    
    
    def res_conv(self,x, s, filters):
        '''
        here the input size changes, when it goes via conv blocks
        so the skip connection uses a projection (conv layer) matrix
        ''' 
        x_skip = x
        f1, f2 = filters
        
        # first block
        x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x)
        # when s = 2 then it is like downsizing the feature map
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)
        
        # second block
        x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)
        
        #third block
        x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        
        # shortcut 
        x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x_skip)
        x_skip = BatchNormalization()(x_skip)
        
        # add 
        x = Add()([x, x_skip])
        x = Activation(activations.relu)(x)
        
        return x
        
        ### Combine the above functions to build 50 layers resnet. 
    def resnet50(self):
        
        input_im = Input(shape=(self.train_images.shape[1], self.train_images.shape[2], self.train_images.shape[3])) # cifar 10 images size
        x = ZeroPadding2D(padding=(3, 3))(input_im)
        
        # 1st stage
        # here we perform maxpooling, see the figure above
        
        x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        
        #2nd stage 
        # frm here on only conv block and identity block, no pooling
        
        x = self.res_conv(x, s=1, filters=(64, 256))
        x = self.res_identity(x, filters=(64, 256))
        x = self.res_identity(x, filters=(64, 256))
        
        # 3rd stage
        
        x = self.res_conv(x, s=2, filters=(128, 512))
        x = self.res_identity(x, filters=(128, 512))
        x = self.res_identity(x, filters=(128, 512))
        x = self.res_identity(x, filters=(128, 512))
        
        # 4th stage
        
        x = self.res_conv(x, s=2, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        
        # 5th stage
        
        x = self.res_conv(x, s=2, filters=(512, 2048))
        x = self.res_identity(x, filters=(512, 2048))
        x = self.res_identity(x, filters=(512, 2048))
        
        # ends with average pooling and dense connection
        
        x = AveragePooling2D((2, 2), padding='same')(x)
        
        x = Flatten()(x)
        x = Dense(len(self.class_types), activation='softmax', kernel_initializer='he_normal')(x) #multi-class
        
        # define the model 
        
        model = Model(inputs=input_im, outputs=x, name='Resnet50')
        
        return model
    
    ### Define some Callbacks
    def lrdecay(self,epoch):
        lr = 1e-3
        if epoch > 180:
            lr *= 0.5e-3
        elif epoch > 160:
            lr *= 1e-3
        elif epoch > 120:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1
        #print('Learning rate: ', lr)
        return lr
      # if epoch < 40:
      #   return 0.01
      # else:
      #   return 0.01 * np.math.exp(0.03 * (40 - epoch))
    
    
    def earlystop(self,mode):
        if mode=='acc':
            estop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=15, mode='max')
        elif mode=='loss':
            estop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, mode='min')
        return estop    
    
    def train_resnet50(self):
        #(self.train_images, self.train_labels), (self.test_images, self.test_labels) = datasets.cifar10.load_data()
            #### Normalize the images to pixel values (0, 1)
        self.train_images, self.test_images = self.train_images/255.0 , self.test_images/255.0
        #### Check the format of the data 
        print ("train_im, train_lab types: ", type(self.train_images), type(self.train_labels))
        #### check the shape of the data
        print ("shape of images and labels array: ", self.train_images.shape, self.train_labels.shape) 
        print ("shape of images and labels array ; test: ", self.test_images.shape, self.test_labels.shape)
        
        
        #### Check the distribution of unique elements 
        (unique, counts) = np.unique(self.train_labels, return_counts=True)
    
        frequencies = np.asarray((unique, counts)).T
        print (frequencies)
        print (len(unique))
    
        plt.figure(figsize=(10,10))
        for i in range(12):
            plt.subplot(4,3,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.train_images[i], cmap='gray')
            plt.xlabel(self.class_types[self.train_labels[i][0]], fontsize=13)
        plt.tight_layout()    
        plt.show()
        ### One hot encoding for labels 
        self.train_labels_categorical = tf.keras.utils.to_categorical(self.train_labels, num_classes=10, dtype='uint8')
        self.test_labels_categorical = tf.keras.utils.to_categorical(self.test_labels, num_classes=10, dtype='uint8')
        
        
        ### Train -test split 
        self.train_images, valid_images, self.train_labels, valid_labels = train_test_split(self.train_images, self.train_labels_categorical, test_size=0.20, 
                                                                stratify=self.train_labels_categorical, 
                                                                random_state=40, shuffle = True)
    
        print ("train data shape after the split: ", self.train_images.shape)
        print ('new validation data shape: ', valid_images.shape)
        print ("validation labels shape: ", valid_labels.shape)
        
        ##### Include Little Data Augmentation 
        batch_size = 64 # try several values
        train_DataGen = tf.keras.preprocessing.image.ImageDataGenerator(zoom_range=0.2, 
                                                                    width_shift_range=0.1, 
                                                                    height_shift_range = 0.1, 
                                                                    horizontal_flip=True)
        valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
        train_set_conv = train_DataGen.flow(self.train_images, self.train_labels, batch_size=batch_size) # train_lab is categorical 
        valid_set_conv = valid_datagen.flow(valid_images, valid_labels, batch_size=batch_size) # so as valid_lab 
                    
        
        resnet50_model = self.resnet50()
        resnet50_model.summary()
        resnet50_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-3), 
                           metrics=['acc'])
        batch_size=64 # test with 64, 128, 256
          
        lrdecay_var = tf.keras.callbacks.LearningRateScheduler(self.lrdecay) # learning rate decay  
        estop_var = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=15, mode='max')
        resnet_train = resnet50_model.fit(train_set_conv, 
                                      epochs=1, 
                                      steps_per_epoch=self.train_images.shape[0]/batch_size, 
                                      validation_steps=valid_images.shape[0]/batch_size, 
                                      validation_data=valid_set_conv, 
                                      callbacks=[lrdecay_var,estop_var])
        return resnet50_model
        
if __name__ == '__main__':
#    model = cnn() 
    TrainingModel = DeepLearningNetworks()
    model = TrainingModel.start_cnn()
#    model = TrainingModel.train_resnet50()
 
    tf.saved_model.save(model, "./models/simple_model")    
    
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
                      logdir="./frozen_models",
                      name="frozen_graph.pb",
                      as_text=False)
        
    
