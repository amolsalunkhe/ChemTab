# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 22:13:59 2021

@author: amol
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.callbacks import EarlyStopping
# patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=45)

from tensorflow.keras import regularizers, activations, initializers, constraints, Sequential
from tensorflow.keras.constraints import UnitNorm, Constraint
from tensorflow.keras import backend as K
from .dnnmodel_model_factory import DNNModelFactory

class SimpleDNNModelFactory(DNNModelFactory):    
    def __init__(self):
        #print("SimpleDNNModel Instantiated")
        self.setModelName("SimpleDNNModel")
        self.setConcreteClassCustomObject({"SimpleDNNModelFactory": SimpleDNNModelFactory})
        return
    
    def build_and_compile_model(self,noOfInputNeurons):
        inputs = keras.Input(shape=(noOfInputNeurons,), name="inputs")
        x = self.getIntermediateLayers(inputs)
        souener_pred = layers.Dense(1, name="prediction")(x)

        model = keras.Model(
            inputs=[inputs],
            outputs=[souener_pred],
        )

        opt = self.getOptimizer()
        
        model.compile(loss='mean_absolute_error',optimizer=opt)
        
        self.model = model
        
        return model
