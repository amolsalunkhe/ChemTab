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

    # reimplmeneted to non-trivial version in PCDNNv2
    # this only exists for compatibility reasons really it should only be called by addLinearLayer()
    def getLinearLayer(self,noOfInputNeurons, noOfCpv, **kwds):
        layer = layers.Dense(noOfCpv, name="linear_embedding", activation="linear")
        return layer

    def addLinearModel(self, inputs, noOfInputNeurons, noOfCpv, concatenateZmix='N', **kwds):
        """ adds PCA linear embedding 'model' (as a layer) 
            to the input tensors inside 'inputs' arg 
            (**kwds are to be passed to getLinearLaye()) """

        # make into boolean
        concatenateZmix=concatenateZmix=='Y'

        # the [1] is really important because that skips the extra (batch)
        # dimension that keras adds implicitly
        assert noOfInputNeurons == inputs[0].shape[1]
        output = self.getLinearLayer(noOfInputNeurons, noOfCpv, **kwds)(inputs[0])

        # implicitly if there are 2 input layers then we want to add mix
        assert concatenateZmix == (len(inputs)>1)
        if concatenateZmix:
            zmix = inputs[1]
 
            #Concatenate the Linear Embedding and Zmix together
            output = layers.Concatenate(name="concatenated_zmix_linear_embedding")([zmix, output])

        linear_emb_model = keras.models.Model(inputs=inputs, outputs=output, name="linear_embedding")
        return linear_emb_model(inputs)

    def addRegressorModel(self, x):
        """Gets layers for regression module of model (renamed from get intermediate layers)"""
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

        # the [1:] is really important because that removes the extra (batch)
        # dimension that keras adds implicitly
        input_ = layers.Input(x.shape[1:])
    
        output = add_regularized_dense_module(input_, [32,64,128])
        output = add_regularized_dense_module(output, [256,512,256])
        output = add_regularized_dense_module(output, [128,64,32])

        # used to be named 'prediction' (now model is named 'prediction', since it is last layer)
        souener_pred = layers.Dense(1)(output)
        regressor_model=keras.models.Model(inputs=input_, outputs=souener_pred, name='prediction')

        return regressor_model(x)
 
    def saveCurrModelAsBestModel(self):
        #print("current directory " + os.getcwd())

        import os
        os.mkdir('./models/best_models/')
        
        # open a file, where you ant to store the data
        file = open("./models/best_models/"+self.modelName+"_experimentSettings", "wb")
        
        # dump information to that file
        pickle.dump(self.experimentSettings, file)
        
        # close the file
        file.close()
        
        #self.model.save("models\\best_models\\"+self.modelName)
        filePath = "./models/best_models/"+self.modelName+".h5"
        tf.keras.models.save_model(self.model, filePath, overwrite=True, include_optimizer=False, save_format='h5')
        
        
    def openBestModel(self):
        #print("current directory" + os.getcwd())
        filePath = "./models/best_models/"+self.modelName+".h5"
        self.model = tf.keras.models.load_model(filePath, custom_objects=self.concreteClassCustomObject)
        
        # open a file, where you stored the pickled data
        file = open("./models/best_models/"+self.modelName+"_experimentSettings", "rb")
        
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
        model = None
        
        species_layer = None
        
        zmix_input = None
        
        linear_embedding_input = None
        
        concatenated_zmix_linear_embedding_layer = None
        
        for layer in self.model.layers:
            print(layer.name) 
            if layer.name == "species_input":
                species_layer = layer
            elif layer.name == "prediction":
                prediction_layer = layer
            elif layer.name == "zmix":
                zmix_input =  keras.Input(shape=(layer._keras_shape,), name="zmix_input")
            elif layer.name == "linear_embedding":
                linear_embedding_input = keras.Input(shape=(layer._keras_shape,), name="linear_embedding_input")        
            elif layer.name == "concatenated_zmix_linear_embedding":    
                x = layers.Concatenate(name="concatenated_zmix_linear_embedding")([zmix_input, linear_embedding_input])
            else:
                x = x
            
        if zmix_layer is not None:
            model = tf.keras.Model ([concatenated_zmix_linear_embedding_layer.input],[prediction_layer.output])
            
        else:
           model = tf.keras.Model ([linear_embedding_layer.input],[prediction_layer.output])
        
        '''
        self.model.layers.pop(0)
        self.model.summary()
        '''
        
        model.summary()

        return model
