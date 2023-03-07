import torch as th
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import pandas as pd

class UQMomentsDataset(Dataset):
    """ Generic UQ Dataset which uses split-apply-combine on group_key to Produce UQ moments (mean & variance)"""
    def __init__(self, csv_fn, inputs_like=None, outputs_like=None, group_key='flame_key'):
        outs_df = inputs_df = df = pd.read_csv(csv_fn)
        df.index=df[group_key]
        if inputs_like: inputs_df = df.filter(like=inputs_like)
        if outputs_like: outs_df = df.filter(like=outputs_like)
        self.df_mu = th.Tensor(inputs_df.groupby(group_key).mean().values)
        self.df_sigma = th.Tensor(inputs_df.groupby(group_key).std().values)
        self.outs_df = th.Tensor(outs_df.groupby(group_key).mean().values)

        print('original df len: ', len(df))
        print('reduced df len: ', self.df_sigma.shape[0])

    def __len__(self):
        return self.df_mu.shape[0]

    def __getitem__(self, idx):
        #inputs = th.randn((self.df_mu.shape[1]),)*self.df_sigma[idx,:] + self.df_mu[idx,:]
        outputs = self.outs_df[idx,:]
        return (self.df_mu[idx,:], self.df_sigma[idx,:]), outputs


class UQSamplesDataset(Dataset):
    """ Wrapper for UQMomentsDataset which produces samples from the corresponding moments """
    def __init__(self, moments_dataset):
        self.moments_dataset = moments_dataset

    def __len__(self):
        return len(self.moments_dataset)

    def __getitem__(self, idx):
        (mu, sigma), outputs = self.moments_dataset[idx]
        inputs = th.randn(*mu.shape)*sigma + mu
        return inputs, outputs

class UQErrorPredictionDataset(Dataset):
    def __init__(self, target_model, moments_dataset):
        self.target_model = target_model
        self.moments_dataset = moments_dataset
        self.sampling_dataset = UQSamplesDataset(moments_dataset)

        input_samples, outputs = self.sampling_dataset[:]
        preds = self.target_model(input_samples)
        self.abs_err_model = th.abs(preds-outputs).detach()

    def __len__(self):
        return len(self.moments_dataset)
    
    def __getitem__(self, index):
        (mu, sigma), outputs = self.moments_dataset[index]
        #input_samples, outputs = self.sampling_dataset[index]
        #preds = self.target_model(input_samples.view(1,-1))
        #abs_err = th.abs(preds-outputs)
        return  th.cat((mu, sigma), axis=-1), self.abs_err_model[index]

class UQModel(pl.LightningModule):
    def __init__(self, input_size=53, output_size=None, n_layers=5, lr=1e-8):
        super().__init__()
        if not output_size: output_size = input_size
        linear_selu = [nn.SELU(), nn.Linear(output_size,output_size)]
        self.regressor = nn.Sequential(nn.BatchNorm1d(input_size),nn.Linear(input_size,output_size),*(linear_selu*n_layers))
        self.lr=lr
        
    def forward(self, inputs):
        return self.regressor(inputs)

    def configure_optimizers(self):
        return th.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, training_batch, batch_id, log=True):
        X, Y = training_batch
        Y_pred = self.forward(X)
        loss = F.mse_loss(Y_pred, Y)
        if log: self.log('mse_loss', loss)
        return loss

    # reuse training_step(), but log validation loss
    def validation_step(self, val_batch, batch_id):
        loss = self.training_step(val_batch, batch_id, log=False)
        self.log('val_mse_loss', loss)

import os

def make_data_loaders(dataset, batch_size=32, train_portion=0.8, workers=8):
    train, val = random_split(dataset, [train_portion, 1-train_portion])
    train_loader = DataLoader(train, batch_size=batch_size, num_workers=workers, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, num_workers=workers)
    return train_loader, val_loader

def fit_UQ_model(dataset, name, max_epochs=1, **kwd_args):
    train_loader, val_loader = make_data_loaders(dataset, **kwd_args)
    x,y=next(iter(val_loader)) # hack to find dataset input size!
    mean_regressor = UQModel(input_size=x.shape[1], output_size=y.shape[1])
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator='gpu', devices=1)
    trainer.fit(mean_regressor, train_loader, val_loader)
    with open(f'{name}.pt', 'wb') as f:
        th.save(mean_regressor, f)
    print(f'done fitting {name}!')
    return mean_regressor

if __name__=='__main__':
    ##################### Fit Mean Regressor: #####################
    df_fn = f'{os.environ["HOME"]}/chrest_head.csv'
    moments_dataset = UQMomentsDataset(df_fn, inputs_like='Yi', outputs_like='souspec', group_key='group')
    samples_dataset = UQSamplesDataset(moments_dataset)
    mean_regressor = fit_UQ_model(samples_dataset, 'mean_regressor', max_epochs=100000, batch_size=128, workers=8)
    #########################################################################
    
    ##################### Fit Standard Deviation Regressor: #####################
    STD_dataset = UQErrorPredictionDataset(mean_regressor, moments_dataset)
    mean_regressor = fit_UQ_model(STD_dataset, 'std_regressor', max_epochs=100000, batch_size=128, workers=8)
    #########################################################################


    
