# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 21:05:29 2021

@author: amol
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import os
# patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=45)


class DNNModelFactory:
    def __init__(self):
        self.width = 512
        self.halfwidth = 128
        self.model = None
        self.experimentSettings = None
        self.modelName = None
        self.concreteClassCustomObject = None
        self.regressorLayerNamePrefix = "regressor_"
        print("Parent DNNModelFactory Instantiated")
    
    def setDataSetMethod(self,dataSetMethod):
        self.dataSetMethod = dataSetMethod

    def setModelName(self,modelName):
        self.modelName = modelName
    
    def setConcreteClassCustomObject(self, concreteClassCustomObject):
        self.concreteClassCustomObject = concreteClassCustomObject
        
    def getOptimizer(self):
        starter_learning_rate = 0.1
        end_learning_rate = 0.01
        decay_steps = 10000
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(starter_learning_rate, decay_steps, end_learning_rate, power=0.5)
        
        #opt = keras.optimizers.Adam(learning_rate=0.001)
        
        opt = keras.optimizers.Adam(learning_rate=learning_rate_fn)
                
        return opt
    
    def getIntermediateLayers(self,x):
        layerNamePrefix = "regressor_" #self.regressorLayerNamePrefix
        
        '''
        cnt = 0
        x = layers.Dense(32,name=layerNamePrefix+str(cnt), activation="relu")(x)
        
        cnt = cnt + 1
        x = layers.Dense(64,name=layerNamePrefix+str(cnt), activation="relu")(x)
        cnt = cnt + 1
        x = layers.Dense(128,name=layerNamePrefix+str(cnt), activation="relu")(x)
        cnt = cnt + 1
        x = layers.Dense(256,name=layerNamePrefix+str(cnt), activation="relu")(x)
        cnt = cnt + 1
        x = layers.Dense(512,name=layerNamePrefix+str(cnt), activation="relu")(x)
        cnt = cnt + 1
        x = layers.Dropout(0.5,noise_shape=None, seed=None, name=layerNamePrefix+str(cnt))(x)
        cnt = cnt + 1
        x = layers.Dense(256,name=layerNamePrefix+str(cnt), activation="relu")(x)
        cnt = cnt + 1
        x = layers.Dense(128,name=layerNamePrefix+str(cnt), activation="relu")(x)
        cnt = cnt + 1
        x = layers.Dense(64,name=layerNamePrefix+str(cnt), activation="relu")(x)
        cnt = cnt + 1
        x = layers.Dense(32,name=layerNamePrefix+str(cnt), activation="relu")(x)
        return x
        '''
        
        def add_regularized_dense_layer(x, layer_size, activation_func='relu', dropout_rate=0.25):

            x = layers.Dense(layer_size, activation=activation_func)(x)

            x = layers.BatchNormalization()(x)

            x = layers.Dropout(dropout_rate)(x)

            return x

        def add_regularized_dense_module(x, layer_sizes, activation_func='relu', dropout_rate=0.25):

            assert len(layer_sizes)==3

            skip_input = x = add_regularized_dense_layer(x, layer_sizes[0], activation_func=activation_func, dropout_rate=dropout_rate)

            x = add_regularized_dense_layer(x, layer_sizes[1], activation_func=activation_func, dropout_rate=dropout_rate)

            x = add_regularized_dense_layer(x, layer_sizes[2], activation_func=activation_func, dropout_rate=dropout_rate)

            x = layers.Concatenate()([x, skip_input])

            return x

        x = add_regularized_dense_module(x, [32,64,128])

        x = add_regularized_dense_module(x, [256,512,256])

        x = add_regularized_dense_module(x, [128,64,32])


        return x


    
    def saveCurrModelAsBestModel(self):
        #print("current directory " + os.getcwd())
        
        # open a file, where you ant to store the data
        file = open("models\\best_models\\"+self.modelName+"_experimentSettings", "wb")
        
        # dump information to that file
        pickle.dump(self.experimentSettings, file)
        
        # close the file
        file.close()
        
        #self.model.save("models\\best_models\\"+self.modelName)
        filePath = "models\\best_models\\"+self.modelName+".h5"
        tf.keras.models.save_model(self.model, filePath, overwrite=True, include_optimizer=False, save_format='h5')
        
        
    def openBestModel(self):
        #print("current directory" + os.getcwd())
        filePath = "models\\best_models\\"+self.modelName+".h5"
        self.model = tf.keras.models.load_model(filePath, custom_objects=self.concreteClassCustomObject)
        
        # open a file, where you stored the pickled data
        file = open("models\\best_models\\"+self.modelName+"_experimentSettings", "rb")
        
        # dump information to that file
        self.experimentSettings = pickle.load(file)
        
        # close the file
        file.close()
        
        return self.model, self.experimentSettings 
    
    def getLinearEncoder(self):
        
        input_layer = None
        
        linear_embedding_layer = None
        
        zmix_layer =  None
        
        for layer in self.model.layers:
            print(layer.name)            
            if layer.name == "species_input":
                input_layer = layer
            if layer.name == "zmix":
                zmix_layer = layer
            if layer.name == "linear_embedding":
                linear_embedding_layer = layer
        
        if zmix_layer is not None:
            model = tf.keras.Model ([input_layer.input],[linear_embedding_layer.output])
            
        else:
           model = tf.keras.Model ([input_layer.input],[linear_embedding_layer.output])
        
        model.summary()

        return model
    
    def getRegressor(self):

        #Copy the model configuration
        model_cfg = self.model.get_config()
       
        
        
        model_cfg['layers'][0] = {
                      'class_name': 'InputLayer',
                      'config': {
                          'batch_input_shape': (None, 2),
                          'dtype': 'float32',
                          'sparse': False,
                          'ragged': False,
                          'name': 'linear_embedding'
                      },
                      'name': 'linear_embedding',       
                      'inbound_nodes': []
                  }
        
        model_cfg['layers'].pop(2)
        
        model_cfg['input_layers'] = [['linear_embedding', 0, 0], ['zmix', 0, 0]]
 
        regressor = tf.keras.Model().from_config(model_cfg,custom_objects=self.concreteClassCustomObject)        
       
        regressor.summary() 
        
        #Copy the Weights of the Layers
        weights = [layer.get_weights() for layer in self.model.layers[3:]]
        
        for layer, weight in zip(regressor.layers[2:], weights):
            layer.set_weights(weight)
       
        
        return regressor