# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 19:42:05 2021

@author: amol
"""
import random
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import *
from sklearn.utils import shuffle


class PositiveLogNormalCol:
    def __init__(self):
        self.min_value = None
        return

    def fit_transform(self, data):
        temp = pd.DataFrame(data=data.astype('float64'), columns=["target"])
        if self.min_value is None:
            self.min_value = temp["target"].min()
            assert self.min_value is not None

            # 2*self.set_max_value --> to account for max that may be out of this dataset
        temp['transfomed'] = temp.apply(lambda row: np.log1p((row.target - self.min_value)), axis=1)
        transfomed_data = temp['transfomed'].values
        transfomed_data = transfomed_data.reshape(transfomed_data.shape[0], 1)
        return transfomed_data

    def inverse_transform(self, transformeddata):
        # todo: complete this
        temp = pd.DataFrame(data=transformeddata.astype('float64'), columns=["target"])
        temp['inverse'] = temp.apply(lambda row: np.expm1((row.target)) + self.min_value, axis=1)
        data = temp['inverse'].values
        try:
            assert (data != float('inf')).all() and (data != -float('inf')).all()
        except AssertionError:
            import pdb;
            pdb.set_trace()
        data = data.reshape(data.shape[0], 1)
        return data


class PositiveLogNormal:
    def __init__(self):
        self.log_col_transformers = None

    def fit_transform(self, data):
        data = np.asarray(data)
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        self.log_col_transformers = []
        cols_data = []
        for i in range(data.shape[1]):
            try:
                self.log_col_transformers.append(PositiveLogNormalCol())
                cols_data.append(self.log_col_transformers[i].fit_transform(data[:, i]))
            except TypeError as e:
                import pdb;
                pdb.set_trace()
                raise e

        cols_data = np.concatenate(cols_data, axis=1)
        return cols_data

    def inverse_transform(self, data):
        cliped = np.clip(data, -708, 708)
        if np.any(data != cliped):
            warnings.warn(
                'clipping data to avoid np.exp overflow, this will cause an imperfect inversion of the transformation')
        data = cliped
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        cols_data = []
        for i in range(data.shape[1]):
            cols_data.append(self.log_col_transformers[i].inverse_transform(data[:, i]))
        return np.concatenate(cols_data, axis=1)  # .squeeze()

def get_df_IQR(df):
	df = pd.DataFrame(df)
	q1 = df.quantile(.25)
	q3 = df.quantile(.75)
	return q3-q1

class DataManager:
    def __init__(self, df_totalData, constants):
        # Dwyer: I assume this is the case if its not update: includePCDNNV2_PCA_data
        assert constants.getDataframe() is df_totalData
        self.constants = constants
        self.df = df_totalData
        self.df['souspecAR'] = 0  # dummy value since it is constant

        global all_dependants
        #all_dependants = ["souener", "souspecO2", "souspecCO", "souspecCO2", "souspecH2O", "souspecOH", "souspecH2", "souspecCH4"]
        all_dependants = ['souener'] + [col for col in self.df.columns if col.startswith('Yi')]
        #all_dependants = [col for col in self.df.columns if col.startswith('Yi')]

        self.outputScaler = None
        self.inputScaler = None
        self.zmixScaler = None
        self.sourceScaler = None
        self.df_training = None
        self.df_testing = None
        self.X_train = None
        self.X_test = None
        self.X_scaled_train = None
        self.X_scaled_test = None
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
        self.dataSetMethod = None
        self.train_portion = 0.5

    def include_PCDNNV2_PCA_data(self, PCDNNV2_model_factory, concatenateZmix: str):
        self.constants.include_PCDNNV2_PCA_data(self, PCDNNV2_model_factory, concatenateZmix)
        self.df = self.constants.getDataframe()  # maybe not necessary...?

    def save_PCA_data(self, fn='PCA_data.csv'):
        self.df.to_csv(fn, index=False)

    @staticmethod
    def train_test_split_on_flamekey(df, train_portion=0.5, seed=0):
        random.seed(seed)
        flame_keys = list(set(df['flame_key_int']))
        random.shuffle(flame_keys)
        train_set_keys = flame_keys[:int(len(flame_keys) * train_portion)]
        test_set_keys = flame_keys[int(len(flame_keys) * train_portion):]
        print('train_set_keys: ', train_set_keys)
        print('test_set_keys: ', test_set_keys)

        train_set = df[np.isin(df['flame_key_int'], train_set_keys)]
        test_set = df[np.isin(df['flame_key_int'], test_set_keys)]
        print('train: ', train_set['flame_key_int'].unique()[:5])
        print('test: ', test_set['flame_key_int'].unique()[:5])
        return train_set, test_set

    def _createTrainTestDfs(self, method):
        if method == 'randomequalflamesplit':
            df_shuffled = shuffle(self.df, random_state=0)
            self.df_training, self.df_testing = self.train_test_split_on_flamekey(df_shuffled,
                                                                                  train_portion=self.train_portion)
        elif (method == "randomequaltraintestsplit"):
            df_shuffled = shuffle(self.df, random_state=0)
            self.df_training = df_shuffled.sample(frac=self.train_portion, random_state=0)
            self.df_testing = df_shuffled.drop(index=self.df_training.index)
        else:
            if self.train_portion is not None:
                raise NotImplementedError(
                    'custom train/test split is only supported for flame-keys and point-based splitting')
            training_flames_int = []
            testing_flames_int = []

            if (method == "frameworkincludedexcludedequalsplit"):

                for x in self.constants.framework_included_flames_int:
                    training_flames_int.append(x)

                for x in self.constants.framework_excluded_flames_int[::2]:
                    training_flames_int.append(x)

                for x in self.constants.framework_included_flames_int:
                    testing_flames_int.append(x)

                for x in self.constants.framework_excluded_flames_int[1::2]:
                    testing_flames_int.append(x)

            elif (method == "frameworkincludedtrainexcludedtest"):
                for x in self.constants.framework_included_flames_int:
                    training_flames_int.append(x)

                for x in self.constants.framework_excluded_flames_int:
                    testing_flames_int.append(x)

            elif (method == "frameworkincludedtrainexcludedandincludedtest"):
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
           
        print('len testing df: ', len(self.df_testing), flush=True)

    def _createTrainTestData(self, method, numCpvComponents):
        method_parts = method.split('_')

        self._createTrainTestDfs(method_parts[1])

        dependants = ["souener"]
        if len(method_parts) > 2 and method_parts[2] == 'AllDependants':
            dependants = all_dependants  # includes souener
        # this is only change for depedendents 
        output_data_cols = dependants

        if method_parts[0] == "ZmixCpv":
            input_data_cols = ["Zmix", "Cpv"]
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
        elif method_parts[0] == "AllSpeciesZmixPCA":
            input_data_cols = self.constants.icovariates
        elif method_parts[0] == "AllSpeciesPurePCA":
            input_data_cols = self.constants.icovariates
        elif method_parts[0] == "AllSpeciesZmixAndPurePCA":
            input_data_cols = self.constants.icovariates
        elif method_parts[0] == "AllSpeciesSparsePCA":
            input_data_cols = self.constants.icovariates
        elif method_parts[0] == "AllSpeciesZmixAndSparsePCA":
            input_data_cols = self.constants.icovariates
        else:
            input_data_cols = self.constants.icovariates

        self.X_train = self.df_training[input_data_cols].values
        self.X_test = self.df_testing[input_data_cols].values
        self.zmix_train = self.df_training['Zmix'].values
        self.zmix_test = self.df_testing['Zmix'].values
        self.Y_train = self.df_training[output_data_cols].values
        self.Y_test = self.df_testing[output_data_cols].values

        self.other_tracking_cols_train = self.df_training[self.constants.other_tracking_cols].values
        self.other_tracking_cols_test = self.df_testing[self.constants.other_tracking_cols].values

        # print("In _createTrainTestData Y_test.shape: " + str(self.Y_test.shape))

        self.input_data_cols = input_data_cols
        self.output_data_cols = output_data_cols

        # column ends used later to identify corresponding source terms
        Yi_cols = [col for col in self.df.columns if col[:2] == 'Yi']
        self.source_input_cols = ['souspec' + col[2:] for col in Yi_cols]
        self.source_train = self.df_training[self.source_input_cols]
        self.source_test = self.df_testing[self.source_input_cols]

        assert len(self.Y_train.shape) == len(self.Y_test.shape)
        if len(self.Y_train.shape) == 1:
            self.Y_train = self.Y_train.reshape(-1, 1)
            self.Y_test = self.Y_test.reshape(-1, 1)

    @property
    def souener_index(self):
        import pdb; pdb.set_trace()
        souener_index = self.output_data_cols.index('souener')
        assert souener_index == 0  # souener_index MUST BE 0 this is an assumption made in other parts of the code too
        # DON'T REMOVE THIS ASSERT!     
        return souener_index

    def _setInputOutputScalers(self, ipscaler, opscaler):
        scalers = {None: lambda: None, 'MinMaxScaler': MinMaxScaler, 'MaxAbsScaler': MaxAbsScaler,
                   'StandardScaler': StandardScaler, 'RobustScaler': RobustScaler,
                   'QuantileTransformer': QuantileTransformer, 'PositiveLogNormal': PositiveLogNormal}

        self.inputScaler = scalers[ipscaler]()
        self.zmixScaler = scalers[ipscaler]()
        self.sourceScaler = self.inputScaler  # scalers[ipscaler]()
        # transformation much match exactly across sources and species

        self.outputScaler = scalers[opscaler]()

    def getSourceTrainTestData(self):
        if self.sourceScaler:
            return self.sourceScaler.transform(self.source_train), self.sourceScaler.transform(self.source_test)
        else:
            return self.source_train, self.source_test

    # TODO: MULTIOUTPUTS, add dependents argument
    def createTrainTestData(self, dataSetMethod, numCpvComponents, ipscaler, opscaler):
        self.dataSetMethod = dataSetMethod
        self._createTrainTestData(dataSetMethod, numCpvComponents)

        # perform scaling 
        self._setInputOutputScalers(ipscaler, opscaler)

        if self.inputScaler is not None:
            #self.inputScaler.fit(np.concatenate((self.X_train, self.X_test), axis=0))
            self.inputScaler.fit(np.concatenate((self.source_train, self.source_test), axis=0))
            #self.inputScaler.fit(np.concatenate((self.X_train, self.X_test, self.source_train, self.source_test), axis=0))
            self.X_scaled_train = self.inputScaler.transform(self.X_train)
            self.X_scaled_test = self.inputScaler.transform(self.X_test)

            self.zmix_scaled_train = self.zmixScaler.fit_transform(self.zmix_train.reshape(self.zmix_train.shape[0], 1))
            self.zmix_scaled_test = self.zmixScaler.transform(self.zmix_test.reshape(self.zmix_test.shape[0], 1))
        else:
            self.X_scaled_train = None
            self.X_scaled_test = None
            self.zmix_scaled_train = None
            self.zmix_scaled_test = None

        if self.outputScaler is not None:
            ## apply to everything
            #self.outputScaler.fit(np.concatenate((self.Y_train, self.Y_test),axis=0))
            #self.Y_scaled_train = self.outputScaler.transform(self.Y_train)
            #self.Y_scaled_test = self.outputScaler.transform(self.Y_test)

            # only apply to source energy
            self.Y_scaled_train = np.copy(self.Y_train)
            self.Y_scaled_test = np.copy(self.Y_test)
            self.outputScaler.fit(np.concatenate((self.Y_train[:,0:1], self.Y_test[:,0:1]),axis=0))
            self.Y_scaled_train[:,0:1] = self.outputScaler.transform(self.Y_train[:,0:1])
            self.Y_scaled_test[:,0:1] = self.outputScaler.transform(self.Y_test[:,0:1])
        else:
            self.Y_scaled_train = None
            self.Y_scaled_test = None
        # print("In createTrainTestData Y_test.shape: " + str(self.Y_test.shape))

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
        else:
            Y_train = self.Y_train
            Y_test = self.Y_test

        return X_train, X_test, Y_train, Y_test, zmix_train, zmix_test

    def getAllData(self):
        X_train, X_test, Y_train, Y_test, zmix_train, zmix_test = self.getTrainTestData()
        source_train, source_test = self.getSourceTrainTestData()

        X = None
        Y = None
        zmix = None
        source = None

        if X_train is not None:
            X = np.append(X_train, X_test, axis=0)
        if Y_train is not None:
            Y = np.append(Y_train, Y_test, axis=0)
        if zmix_train is not None:
            zmix = np.append(zmix_train, zmix_test, axis=0)
        if source_train is not None:
            source = np.append(source_train, source_test, axis=0)

        # master index represents original index of self.df reshuffled for training/testing
        # we can use this to return to original order! (unshuffle)
        master_index = np.append(self.df_training.index, self.df_testing.index)

        def reorder(array):
            if array is not None:
                new_array = array.copy()  # GOTCHA: assignment isn't atomic!
                new_array[master_index] = array  # unshuffle df/array
                array = new_array
            return array

        # this should fix PCA logging bug! 
        return reorder(X), reorder(Y), reorder(zmix), reorder(source)
