# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 17:52:13 2021

@author: amol
"""

import time
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.optimize
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn import gaussian_process
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.utils.optimize import _check_optimize_result
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
from sklearn.decomposition import PCA, SparsePCA
from sklearn.preprocessing import MinMaxScaler
# demonstrate data normalization with sklearn

class DataPreparer:
    def __init__(self):

        
        #read the data into a dataframe
        self.df = pd.read_csv('../NewData_flames_data_with_L1_L2_errors_CH4-AIR_with_trimming.txt')
        self.df = self.df.drop(columns=[' L1_ERR', ' L2_ERR'], errors='ignore')       
 
        self.num_principal_components = 5
        
        #create an integer representation of the flame-id and add to the data frame
        self.df['flame_key_int'] = self.df[' flame_key'].mul(10000000).astype(int)
        
        #create an integer to determine if the flame is included by the framework in the manifold creation and reverselookup
        #framework_untrimmed_flameids = [0.00115982, 0.00122087, 0.00128512, 0.00135276, 0.00142396, 0.0014989, 0.00157779, 0.00166083, 0.00174825, 0.00184026, 0.00193711, 0.00203907, 0.00214639, 0.00225936, 0.00237827, 0.01]
        
        self.framework_untrimmed_flameids = ['2.0276547153583627E-4', '2.1343733845877503E-4', '2.2467088258818426E-4', '2.3649566588229923E-4', '2.4894280619189394E-4', '2.6204505914936203E-4', '2.7583690436774953E-4', '2.903546361765785E-4', '3.056364591332405E-4', '3.2172258856130585E-4', '3.3865535638032194E-4', '0.0032353354497370902']
        
        
        self.framework_untrimmed_flame_key_ints = [int(float(self.framework_untrimmed_flameids[i])*10000000) for i in range(len(self.framework_untrimmed_flameids))]
    
        
        self.df['is_flame_included_by_framework'] = self.df['flame_key_int'].map(lambda x: self.isFlame_included(x))
    
    
        self.df['souener_deciles'] = pd.qcut(self.df['souener'],10)

        self.icovariates = []
        for c in self.df.columns:
            if c[0:2] == 'Yi':
                self.icovariates.append(c)
        
        self.zmix_pca_dim_cols = ["Zmix_PCA_"+str(i+1) for i in range(self.num_principal_components)]    
        
        self.pure_pca_dim_cols = ["PURE_PCA_"+str(i+1) for i in range(self.num_principal_components)]
        
        self.sparse_pca_dim_cols = ["SPARSE_PCA_"+str(i+1) for i in range(self.num_principal_components)]
        
        self.framework_included_flames_int = self.df[self.df['is_flame_included_by_framework'] == 1]['flame_key_int'].unique()

        self.framework_excluded_flames_int = self.df[self.df['is_flame_included_by_framework'] == 0]['flame_key_int'].unique()

        self.all_flames_int = self.df['flame_key_int'].unique()
        
        self.other_tracking_cols = ['is_flame_included_by_framework','Xpos',' flame_key','flame_key_int']
        
    def isFlame_included(self,flame_key_int):
        if flame_key_int in self.framework_untrimmed_flame_key_ints:
            ret_val = 1
        else:
            ret_val = 0
        return ret_val
    
    def include_PCDNNV2_PCA_data(self, dm, model_factory, concatenateZmix: str):
        X_train, X_test, Y_train, Y_test, rom_train, rom_test, zmix_train, zmix_test = dm.getTrainTestData()
        X = np.concatenate((X_train, X_test),axis=0)
        zmix = np.concatenate((zmix_train, zmix_test),axis=0).squeeze()
        Y = np.concatenate((Y_train, Y_test),axis=0).squeeze()
        PCA_model = model_factory.getLinearEncoder()

        inputs = {"species_input":X, "zmix":zmix} if concatenateZmix == 'Y' else {"species_input":X}
        
        PCAs = PCA_model.predict({"species_input":X})
        predictions = model_factory.model.predict(inputs).squeeze()
        
        if dm.outputScaler: # These errors need to be raw 
            predictions = dm.outputScaler.inverse_transform(predictions.reshape(-1, 1)).squeeze()
            Y = dm.outputScaler.inverse_transform(Y.reshape(-1,1)).squeeze()

        #error_df = pd.DataFrame(np.stack((predictions-Y)**2, np.abs(predictions-Y)), columns=['L2_ERR', 'L1_ERR'])
        PCDNNV2_PCA_df = pd.DataFrame(PCAs, columns=[f'PCDNNV2_PC_{i+1}' for i in range(PCAs.shape[1])])
        self.df[PCDNNV2_PCA_df.columns] = PCDNNV2_PCA_df
        self.df['L1_ERR'] = np.abs(predictions-Y)
        self.df['L2_ERR'] = (predictions-Y)**2


    def createPCAs(self):

        pca = PCA(n_components=self.num_principal_components)
    
        X = self.df[self.icovariates].values
        
        pca.fit_transform(X)
                
        df_pure_pca = pd.DataFrame(pca.transform(X), columns = self.pure_pca_dim_cols)
                
        self.df = pd.concat([self.df,df_pure_pca], axis=1)

    def sparsePCAs(self):
        
        sparsepca = SparsePCA(n_components=self.num_principal_components)
      
        X = self.df[self.icovariates].values
        
        sparsepca.fit_transform(X)
                
        df_sparse_pca = pd.DataFrame(sparsepca.transform(X), columns = self.sparse_pca_dim_cols)
                
        self.df = pd.concat([self.df,df_sparse_pca], axis=1)
    
    def zmixOrthogonalPCAs(self):
        X = self.df[self.icovariates].values

        #these are the weights calculated on the basis of molar weight of Hydrogen
        wopt = np.array([0.25131806468584, 1.0, 0.0, 0.0, 0.05926499970012948, 0.11189834407236524, 0.03053739933116691, 0.05926499970012948, 0.0, 0.07742283372149472, 0.14371856860332313, 0.14371856860332313, 0.20112514400193687, 1.0, 0.0, 0.0, 0.03473494419333629, 0.06713785861443991, 0.09743596683886535, 0.09743596683886535, 0.12582790137651187, 0.04027033873046593, 0.07742283372149472, 0.11180607885607882, 0.14371856860332313, 0.17341738612784788, 0.20112514400193687, 0.024566681794273966, 0.04795526192839207, 0.04795526192839207, 0.0, 0.06713048065088474, 0.12581494366075874, 0.17755300484072126, 0.034730994502665966, 0.0, 0.0, 0.0, 0.03249947443158002, 0.0, 0.0372961080230628, 0.07191024382448291, 0.024564706019978535, 0.023426986426879046, 0.023426986426879046, 0.023426986426879046, 0.0, 0.16374935944566987, 0.18286442054789118, 0.07024850027715426, 0.09152158240065958, 0.0, 0.0] , dtype=float)
        
        w = wopt[:,np.newaxis]

        # center the data
        Xcenter = X - np.mean(X)
        
        A = np.cov(X.T)
        
        # calculate A - ww^TA
        L = A - np.dot(np.dot(w,w.T),A)
        
        # get the first eigen vector
        values,vectors = np.linalg.eig(L)
        
        vectors = np.real(vectors)
        
        values = np.real(values)
        
        df_zmix_pca = pd.DataFrame()
        
        '''
        To reproduce Zmix the actual formula should be 
        
        df_zmix_pca[zmix_pca_dim_cols[0]] = X.dot(wopt)/0.25131806468584
        
        instead of
        
        df_zmix_pca[zmix_pca_dim_cols[0]] = Xcenter.dot(wopt)
        '''
        
        df_zmix_pca[self.zmix_pca_dim_cols[0]] = X.dot(wopt)/0.25131806468584
        
        for i in range(len(self.zmix_pca_dim_cols)-1):
            df_zmix_pca[self.zmix_pca_dim_cols[i+1]] = Xcenter.dot(vectors.T[i])
                
        self.df = pd.concat([self.df,df_zmix_pca], axis=1)
    
    def getDataframe(self):
        return self.df
