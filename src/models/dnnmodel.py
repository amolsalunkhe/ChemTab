# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 21:05:29 2021

@author: amol
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
# patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=45)


class DNNModel:
    def __init__(self):
        self.width = 512
        self.halfwidth = 128
        print("Parent DNNModel Instantiated")
    
    def getOptimizer(self):
        starter_learning_rate = 0.1
        end_learning_rate = 0.01
        decay_steps = 10000
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(starter_learning_rate, decay_steps, end_learning_rate, power=0.5)
        
        #opt = keras.optimizers.Adam(learning_rate=0.001)
        
        opt = keras.optimizers.Adam(learning_rate=learning_rate_fn)
                
        return opt
    
    def getIntermediateLayers(self,x):
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dense(32, activation="relu")(x)
        return x
    

    