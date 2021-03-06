# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 19:42:05 2021

@author: amol
"""
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

class PositiveLogNormal:
    def __init__(self):
        self.max_value = 0
        return
    
    def set_max_value(self,max_value):
        self.max_value = max_value
        return 
    
    def get_max_value(self):
        return self.max_value
        
    def fit_transform(self,data):
        temp = pd.DataFrame(data=data, columns=["target"])
        max_value = temp["target"].max()
        self.set_max_value(max_value)
        #2*self.set_max_value --> to account for max that may be out of this dataset
        temp['transfomed'] = temp.apply(lambda row: np.log1p((row.target + 2*self.max_value)), axis=1)
        transfomed_data = temp['transfomed'].values
        transfomed_data = transfomed_data.reshape(transfomed_data.shape[0], 1)
        return transfomed_data
    
    def inverse_transform(self,transformeddata):
        #todo: complete this
        temp = pd.DataFrame(data=transformeddata, columns=["target"])
        temp['inverse'] = temp.apply(lambda row:  np.expm1((row.target)) - 2*self.max_value, axis=1)
        data = temp['inverse'].values
        data = data.reshape(data.shape[0], 1)
        return data

class DataManager:
    def __init__(self, df_totalData, constants):
        self.constants = constants
        self.df = df_totalData
        self.outputScaler = None
        self.inputScaler = None
        self.zmixScaler = None
        self.df_training = None
        self.df_testing = None
        self.X_train = None
        self.X_test = None
        self.X_scaled_train = None
        self.X_scaled_test = None
        self.rom_train = None
        self.rom_test = None
        self.rom_scaled_train = None
        self.rom_scaled_test = None
        self.zmix_train = None
        self.zmix_test = None
        self.zmix_scaled_train = None
        self.zmix_scaled_test = None
        self.Y_train = None
        self.Y_test = None
        self.Y_scaled_train = None
        self.Y_scaled_test = None
        self.input_data_cols = None
        self.output_data_cols = None 
        self.other_tracking_cols_train = None
        self.other_tracking_cols_test = None
        
        return

    def _createTrainTestDfs(self,method):

        if(method == "randomequaltraintestsplit"):
            df_shuffled= shuffle(self.df, random_state=0)
            self.df_training = df_shuffled[::2]
            self.df_testing = df_shuffled[1::2]

        else:
            training_flames_int = []


            testing_flames_int = []

            if(method == "frameworkincludedexcludedequalsplit"):

                for x in self.constants.framework_included_flames_int:
                    training_flames_int.append(x)

                for x in self.constants.framework_excluded_flames_int[::2]:
                    training_flames_int.append(x)

                for x in self.constants.framework_included_flames_int:
                    testing_flames_int.append(x)

                for x in self.constants.framework_excluded_flames_int[1::2]:
                    testing_flames_int.append(x)

            elif(method == "frameworkincludedtrainexcludedtest"):
                for x in self.constants.framework_included_flames_int:
                    training_flames_int.append(x)

                for x in self.constants.framework_excluded_flames_int:
                    testing_flames_int.append(x)

            elif(method == "frameworkincludedtrainexcludedandincludedtest"):
                for x in self.constants.framework_included_flames_int:
                    training_flames_int.append(x)

                for x in self.constants.framework_included_flames_int:
                    testing_flames_int.append(x)

                for x in self.constants.framework_excluded_flames_int:
                    testing_flames_int.append(x)

            else:
                for x in self.constants.all_flames_int:
                    training_flames_int.append(x)
                    testing_flames_int.append(x)

            self.df_training = self.df[self.df['flame_key_int'].isin(training_flames_int)]

            self.df_testing = self.df[self.df['flame_key_int'].isin(testing_flames_int)]

        return


    def _createTrainTestData(self,method,numCpvComponents):

        method_parts = method.split('_')

        self._createTrainTestDfs(method_parts[1])

        input_data_cols = []
        
        rom_cols = []
        
        output_data_cols = ["souener"]        

        if method_parts[0] == "ZmixCpv":
            input_data_cols = ["Zmix","Cpv"]
        elif method_parts[0] == "ZmixPCA":
            input_data_cols = self.constants.zmix_pca_dim_cols[0:numCpvComponents] 
        elif method_parts[0] == "SparsePCA":
            input_data_cols = self.constants.sparse_pca_dim_cols[0:numCpvComponents]
        elif method_parts[0] == "PurePCA":
            input_data_cols = self.constants.pure_pca_dim_cols[0:numCpvComponents]
        elif method_parts[0] == "ZmixAndSparsePCA":
            input_data_cols = ['Zmix'] + self.constants.sparse_pca_dim_cols[0:numCpvComponents]
        elif method_parts[0] == "ZmixAndPurePCA":
            input_data_cols = ['Zmix'] + self.constants.pure_pca_dim_cols[0:numCpvComponents]    
        elif method_parts[0] == "ZmixAllSpecies":    
            input_data_cols = ['Zmix'] + self.constants.icovariates
        elif method_parts[0] == "AllSpeciesZmixCpv":
            input_data_cols = self.constants.icovariates
            rom_cols = ["Cpv"]
        elif method_parts[0] == "AllSpeciesZmixPCA":
            input_data_cols = self.constants.icovariates
            rom_cols = self.constants.zmix_pca_dim_cols[0:numCpvComponents]            
        elif method_parts[0] == "AllSpeciesPurePCA":
            input_data_cols = self.constants.icovariates
            rom_cols = self.constants.pure_pca_dim_cols[0:numCpvComponents]
        elif method_parts[0] == "AllSpeciesZmixAndPurePCA":
            input_data_cols = self.constants.icovariates
            rom_cols = self.constants.pure_pca_dim_cols[0:numCpvComponents]            
        elif method_parts[0] == "AllSpeciesSparsePCA":
            input_data_cols = self.constants.icovariates
            rom_cols = self.constants.sparse_pca_dim_cols[0:numCpvComponents]
        elif method_parts[0] == "AllSpeciesZmixAndSparsePCA":
            input_data_cols = self.constants.icovariates
            rom_cols = self.constants.sparse_pca_dim_cols[0:numCpvComponents]             
        else:
            input_data_cols = self.constants.icovariates

        self.X_train = self.df_training [input_data_cols].values
        self.X_test = self.df_testing [input_data_cols].values
        self.rom_train = self.df_training [rom_cols].values
        self.rom_test = self.df_testing [rom_cols].values
        self.zmix_train = self.df_training ['Zmix'].values
        self.zmix_test = self.df_testing ['Zmix'].values
        self.Y_train = self.df_training [output_data_cols].values
        self.Y_test = self.df_testing [output_data_cols].values
        
        self.other_tracking_cols_train = self.df_training [self.constants.other_tracking_cols].values
        self.other_tracking_cols_test = self.df_testing [self.constants.other_tracking_cols].values
        
        #print("In _createTrainTestData Y_test.shape: " + str(self.Y_test.shape))
       
        self.input_data_cols = input_data_cols
        self.output_data_cols = output_data_cols
 
        return
    
    def _setInputOutputScalers(self, ipscaler, opscaler):
        if ipscaler == "MinMaxScaler":
            self.inputScaler = MinMaxScaler()
            self.zmixScaler = MinMaxScaler()
        elif ipscaler == "QuantileTransformer":
            self.inputScaler = QuantileTransformer()
            self.zmixScaler = QuantileTransformer()
        elif ipscaler == "PositiveLogNormal":
            self.inputScaler = PositiveLogNormal()
            self.zmixScaler = PositiveLogNormal()
        else:
            self.inputScaler = None
            self.zmixScaler = None
        if opscaler == "MinMaxScaler":
            self.outputScaler = MinMaxScaler()
            self.romScaler = MinMaxScaler()
        elif opscaler == "QuantileTransformer":
            self.outputScaler = QuantileTransformer()
            self.romScaler = QuantileTransformer()
        elif opscaler == "PositiveLogNormal":
            self.outputScaler = PositiveLogNormal()
            self.romScaler = PositiveLogNormal()
        else:
            self.outputScaler = None
            self.romScaler = None
            
    def createTrainTestData(self,dataSetMethod,numCpvComponents, ipscaler, opscaler):
        self._createTrainTestData(dataSetMethod,numCpvComponents)
        
        self._setInputOutputScalers(ipscaler, opscaler)
                
        if self.inputScaler is not None:
            self.X_scaled_train = self.inputScaler.fit_transform(self.X_train)
            self.X_scaled_test = self.inputScaler.fit_transform(self.X_test)
            
            self.zmix_scaled_train = self.zmixScaler.fit_transform(self.zmix_train.reshape(self.zmix_train.shape[0], 1))
            self.zmix_scaled_test = self.zmixScaler.fit_transform(self.zmix_test.reshape(self.zmix_test.shape[0], 1))
            self.zmix_train = self.zmix_train.flatten()
            self.zmix_scaled_train = self.zmix_scaled_train.flatten()
            self.zmix_test = self.zmix_test.flatten()
            self.zmix_scaled_test = self.zmix_scaled_test.flatten()
            
        else:
            self.X_scaled_train = None
            self.X_scaled_test = None
            self.zmix_scaled_train = None
            self.zmix_scaled_test = None
            
            
        if self.outputScaler is not None:
            self.Y_scaled_train = self.outputScaler.fit_transform(self.Y_train.reshape(self.Y_train.shape[0], 1))
            self.Y_scaled_test = self.outputScaler.fit_transform(self.Y_test.reshape(self.Y_test.shape[0], 1))
            self.Y_train = self.Y_train.flatten()
            self.Y_scaled_train = self.Y_scaled_train.flatten()
            self.Y_test = self.Y_test.flatten()
            self.Y_scaled_test = self.Y_scaled_test.flatten()
            if not self.rom_train.shape[1] == 0:
                self.rom_scaled_train = self.romScaler.fit_transform(self.rom_train)
                self.rom_scaled_test = self.romScaler.fit_transform(self.rom_test)
        else:
            self.Y_scaled_train = None
            self.Y_scaled_test = None
            self.rom_scaled_train = None
            self.rom_scaled_test = None
        #print("In createTrainTestData Y_test.shape: " + str(self.Y_test.shape))

    def getTrainTestData(self):
        if self.inputScaler is not None:
            X_train = self.X_scaled_train
            X_test = self.X_scaled_test
            zmix_train = self.zmix_scaled_train
            zmix_test = self.zmix_scaled_test
        else:
            X_train = self.X_train
            X_test = self.X_test
            zmix_train = self.zmix_train
            zmix_test = self.zmix_test
        
        if self.outputScaler is not None:
            Y_train = self.Y_scaled_train
            Y_test = self.Y_scaled_test
            rom_train = self.rom_train
            rom_test = self.rom_test
        else:
            Y_train = self.Y_train
            Y_test = self.Y_test
            rom_train = self.rom_scaled_train
            rom_test = self.rom_scaled_test

        return X_train, X_test, Y_train, Y_test, rom_train, rom_test, zmix_train, zmix_test
    
    def getAllData(self):
        X_train, X_test, Y_train, Y_test, rom_train, rom_test, zmix_train, zmix_test = self.getTrainTestData()
        X = None
        Y = None
        rom = None
        zmix = None
        
        if X_train is not None:
            X = np.append(X_train, X_test, axis=0)
        if Y_train is not None:    
            Y = np.append(Y_train, Y_test, axis=0)
        if rom_train is not None:
            rom = np.append(rom_train, rom_test, axis=0)
        if zmix_train is not None:
            zmix = np.append(zmix_train, zmix_test, axis=0)
        
        return X,Y,rom,zmix