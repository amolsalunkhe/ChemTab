# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 17:52:13 2021

@author: amol
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, SparsePCA

# demonstrate data normalization with sklearn

class DataPreparer:
    def __init__(self, fn='../methane_air_master.csv'):
        # read the data into a dataframe
        self.df = pd.read_csv(fn).sort_index(axis=1) # order needs to be sorted for pre-loaded weight matrices
        if 'zmix' in self.df.columns: 
            self.df['Zmix']=self.df['zmix']
            self.df=self.df.drop(columns='zmix')
        #self.df['Zmix'] = self.df['Xpos'] = self.df['X'] = self.df['flame_key'] = 0
       
        self.df.columns = map(lambda x: x.strip(), self.df.columns)  # deal with annoying spaces in column names
        self.df = self.df.drop(columns=['L1_ERR', 'L2_ERR'], errors='ignore')
        # include space misspelling & correct spelling (in-case it is fixed) 

        self.other_tracking_cols=['Xpos', 'flame_key']
        self.other_tracking_cols=[col for col in self.other_tracking_cols if col in self.df.columns]
        self.num_principal_components = 12
        
        self.icovariates = []
        for c in self.df.columns:
            if c[0:2] == 'Yi':
                self.icovariates.append(c)


        ## create an integer representation of the flame-id and add to the data frame
        #self.df['flame_key_int'] = self.df['flame_key'].mul(10000000).astype(int)

        #self.framework_untrimmed_flameids = ['2.0276547153583627E-4', '2.1343733845877503E-4', '2.2467088258818426E-4',
        #                                     '2.3649566588229923E-4', '2.4894280619189394E-4', '2.6204505914936203E-4',
        #                                     '2.7583690436774953E-4', '2.903546361765785E-4', '3.056364591332405E-4',
        #                                     '3.2172258856130585E-4', '3.3865535638032194E-4', '0.0032353354497370902']

        #self.framework_untrimmed_flame_key_ints = [int(float(self.framework_untrimmed_flameids[i]) * 10000000) for i in
        #                                           range(len(self.framework_untrimmed_flameids))]

        #self.df['is_flame_included_by_framework'] = self.df['flame_key_int'].map(lambda x: self.isFlame_included(x))

        #self.icovariates = []
        #for c in self.df.columns:
        #    if c[0:2] == 'Yi':
        #        self.icovariates.append(c)

        #self.zmix_pca_dim_cols = ["Zmix_PCA_" + str(i + 1) for i in range(self.num_principal_components)]

        #self.pure_pca_dim_cols = ["PURE_PCA_" + str(i + 1) for i in range(self.num_principal_components)]

        #self.sparse_pca_dim_cols = ["SPARSE_PCA_" + str(i + 1) for i in range(self.num_principal_components)]

        #self.framework_included_flames_int = self.df[self.df['is_flame_included_by_framework'] == 1][
        #    'flame_key_int'].unique()

        #self.framework_excluded_flames_int = self.df[self.df['is_flame_included_by_framework'] == 0][
        #    'flame_key_int'].unique()

        #self.all_flames_int = self.df['flame_key_int'].unique()

        #self.other_tracking_cols = ['is_flame_included_by_framework', 'Xpos', 'flame_key', 'flame_key_int']

        ##cut_labels = ['0.0 - 0.11', '0.11 - 0.22', '0.22 - 0.33', '0.33 - 0.44', '0.44 - 0.55', '0.55 - 0.66',
        ##              '0.66 - 0.77', '0.77 - 0.88', '0.88 - 0.99', '0.99 - 1.1']
        ##cut_bins = np.linspace(0, 1.1, 11)
        ##self.df['X_bins'] = pd.cut(self.df['X'], bins=cut_bins, labels=cut_labels)

    #def isFlame_included(self, flame_key_int):
    #    if flame_key_int in self.framework_untrimmed_flame_key_ints:
    #        ret_val = 1
    #    else:
    #        ret_val = 0
    #    return ret_val

    def include_PCDNNV2_PCA_data(self, dm, model_factory, concatenateZmix: str):
        # this appends (train, test) data in that order
        X, Y, zmix, sources = dm.getAllData()
        PCA_model = model_factory.getLinearEncoder()

        inputs = {"species_input": X}
        if concatenateZmix == 'Y':
            inputs['zmix'] = zmix

        PCAs = PCA_model.predict({"species_input": X})
        CPV_sources = PCA_model.predict({"species_input": sources})
        all_predictions = model_factory.getEmbRegressor().predict(inputs)
        predictions = all_predictions['static_source_prediction'].squeeze()
        Y = Y.squeeze()  # you get nasty broadcast errors when you don't squeeze Y & predictions!

        if dm.outputScaler:  # These errors need to be raw
            if len(Y.shape) == 1: # its ok notice reshape is behind it, this isn't a bug
                Y = Y.reshape(-1, 1)
                predictions = predictions.reshape(-1, 1)
            predictions = dm.outputScaler.inverse_transform(predictions).squeeze()
            Y = dm.outputScaler.inverse_transform(Y).squeeze()

        # error_df = pd.DataFrame(np.stack((predictions-Y)**2, np.abs(predictions-Y)), columns=['L2_ERR', 'L1_ERR'])
        PCDNNV2_PCA_df = pd.DataFrame(PCAs, columns=[f'PCDNNV2_PCA_{i + 1}' for i in range(PCAs.shape[1])])
        PCDNNV2_PCA_sources_df = pd.DataFrame(CPV_sources, columns=[f'PCDNNV2_PCA_source_{i + 1}' for i in range(PCAs.shape[1])])
        self.df[PCDNNV2_PCA_df.columns] = PCDNNV2_PCA_df
        self.df[PCDNNV2_PCA_sources_df.columns] = PCDNNV2_PCA_sources_df

        L1_ERR = np.abs(predictions - Y)
        L2_ERR = (predictions - Y) ** 2
        if len(L1_ERR.shape) == 1:
            L1_ERR = L1_ERR.reshape(-1, 1)
            L2_ERR = L2_ERR.reshape(-1, 1)

        # We assume & assert elsewhere that index of souener is 0
        # And Amol told me to make error only of souener prediction
        self.df['L1_ERR_souener'] = L1_ERR[:, dm.souener_index]
        self.df['L2_ERR_souener'] = L2_ERR[:, dm.souener_index]
        self.df['L1_ERR_static'] = np.mean(L1_ERR, axis=1)
        self.df['L2_ERR_static'] = np.mean(L2_ERR, axis=1)

        # add dynamic prediction error as well 
        L1_ERR_dynamic = np.abs(all_predictions['dynamic_source_prediction'] - CPV_sources)
        self.df['L1_ERR_dynamic'] = np.mean(L1_ERR_dynamic, axis=1)
        self.df['L2_ERR_dynamic'] = np.mean(L1_ERR_dynamic**2, axis=1)


    #def createPCAs(self):
    #    # necessary to make PCA work properly
    #    from sklearn.preprocessing import StandardScaler
    #    pca = PCA(n_components=self.num_principal_components, whiten=True)

    #    X = self.df[self.icovariates].values
    #    X = StandardScaler().fit_transform(X)

    #    df_pure_pca = pd.DataFrame(pca.fit_transform(X), columns=self.pure_pca_dim_cols)

    #    self.df[self.pure_pca_dim_cols] = df_pure_pca
    #    #self.df = pd.concat([self.df, df_pure_pca], axis=1)

    #def sparsePCAs(self):

    #    sparsepca = SparsePCA(n_components=self.num_principal_components)

    #    X = self.df[self.icovariates].values

    #    sparsepca.fit_transform(X)

    #    df_sparse_pca = pd.DataFrame(sparsepca.transform(X), columns=self.sparse_pca_dim_cols)

    #    self.df = pd.concat([self.df, df_sparse_pca], axis=1)

    def getDataframe(self):
        return self.df
