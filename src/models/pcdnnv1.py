# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 21:57:24 2021

@author: amol
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
# patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=45)
from .dnnmodel import DNNModel

class PCDNNV1Model(DNNModel):
    def __init__(self):
        return

    def build_and_compile_model(self,noOfInputNeurons,noOfCpv,concatenateZmix):

        species_inputs = keras.Input(shape=(noOfInputNeurons,), name="species_input")
        
        linear_reduced_dims = layers.Dense(noOfCpv, name="linear_layer")(species_inputs)

        if concatenateZmix == 'Y':
            zmix = keras.Input(shape=(1,), name="zmix")
            
            x = layers.concatenate([linear_reduced_dims,zmix])
       
            x = self.getIntermediateLayers(x)
            
            inputs = [species_inputs,zmix]
        else:
            x = self.getIntermediateLayers(linear_reduced_dims)
            
            inputs = [species_inputs]
            
        #Predict the source energy
        souener_pred = layers.Dense(1, name="prediction")(x)

        physics_pred = layers.Dense(noOfCpv, name="physics")(linear_reduced_dims)
        
        model = keras.Model(inputs=inputs,outputs=[souener_pred,physics_pred])

        opt = self.getOptimizer()
        
        model.compile(loss={"physics": keras.losses.MeanAbsoluteError(),"prediction": keras.losses.MeanAbsoluteError()},loss_weights=[2.0, 0.2],optimizer=opt)
        
        return model
