library(dplyr)
library(tidyr)
library(purrr)
source('~/.RProfile')

library(h2o)
h2o.init()

fit_aml = function(X_df, Y_vec) {
  df = cbind(X_df, Y_vec)
  colnames(df)[[ncol(df)]]='Y'
  df = as.h2o(df)
  aml <- h2o.automl(x=colnames(X_df), y='Y', training_frame=df,
                    max_models = 15, max_runtime_secs=90, nfolds=5)
  leader = h2o.get_best_model(aml)
  cat('R^2: ', h2o.r2(leader, xval=T))
  return(leader)
}

# NOTE: builtin R^2 is broken for some reason... (not sure why)
get_explained_var_H2O = function(X_df, Y_df, max_models = 15, max_runtime_secs=90) {
  Y_df_prefix = 'Y_' # this fixes issue with duplicate column names across X_df & Y_df
  colnames(Y_df) = paste0(Y_df_prefix, colnames(Y_df))
  df = as.h2o(cbind(X_df, Y_df))
  
  aml_R2 = function(Y_col) { # coupled with outer function
    aml <- h2o.automl(x=colnames(X_df), y=Y_col, training_frame=df,
                      max_models = max_models, max_runtime_secs=max_runtime_secs, nfolds=5)
    leader = h2o.get_best_model(aml)
    return(h2o.r2(leader, xval=T))
  }
  
  library(tidyr)
  library(purrr)
  colnames(Y_df) %>% purrr::map_dbl(~aml_R2(Y_col=.x)) %>% mean(na.rm=T) # why does R^2 get NaNs??
}

PCA_data = read.csv('./datasets/wax_master.csv')# %>% select(-YiN2, -souspecN2) %>% filter(T>350 & abs(souener)>max(souener)*0.01)
mass_frac_data = PCA_data %>% select(starts_with('Yi'))
souspec_data = PCA_data %>% select(starts_with('souspec'))
souspec_data$souspecAR = 0 # doesn't exist because it is just 0, but it is still needed

stopifnot(nrow(PCA_data)>0)
stopifnot(nrow(souspec_data)>0)
#mass_PCs = PCA_data %>% select(matches('Pure_PCA_[0-9]+'))
CPVs = PCA_data %>% select(matches('PCDNNV2_PCA_[0-9]+'))
CPV_sources = PCA_data %>% select(matches('PCDNNV2_PCA_source_[0-9]+'))

n_PCs=10

all_dependants_cols = c("souener", "souspecO2", "souspecCO", "souspecCO2", "souspecH2O", "souspecOH", "souspecH2", "souspecCH4")
all_dependants = PCA_data[,all_dependants_cols]
mass_PCA = stats::prcomp(mass_frac_data, scale.=T, center=T)
#mass_PCs = mass_PCA$x[,1:n_PCs]

# works! wahoo!
stopifnot(all.equal(scale(mass_frac_data)%*%mass_PCA$rotation, mass_PCA$x))

# NOTE: It is ok to ignore bias because entire point is to make this a LINEAR transform
# NOTE: whole point in this is to approximate earlier "proper PCA" with a strictly linear transformation
# Everything is verified except whether it is ok to ignore bias: 8/2/22
fit_linear_transform = function(X_df, Y_df) {
  # verified to work 8/2/22 (checks that data frames are all numeric)
  stopifnot(all(map_lgl(X_df, is.numeric)))
  stopifnot(all(map_lgl(Y_df, is.numeric)))
  
  rotation_matrix = NULL
  model_r2s = NULL
  for (j in 1:ncol(Y_df)) {
    model = lm(Y_df[,j]~.-1, data=X_df) # TODO: verify that it is ok to ignore bias here?
    rotation_matrix = cbind(rotation_matrix, coef(model))
    model_r2s = c(model_r2s, summary(model)$r.squared)
  }
  colnames(rotation_matrix) = colnames(Y_df)
  cat('all model r^2s: ', model_r2s)
  boxplot(model_r2s, main='model R^2s')

  # print R^2 of entire rotation matrix
  MSE = apply((as.matrix(X_df)%*%rotation_matrix-Y_df)**2, -1, mean)
  VAR = apply(Y_df, -1, var)
  R2 = 1-median(MSE/VAR)
  cat('median(R^2) of linear transform fit: ', R2)
  # median avoids numerical errors

  return(rotation_matrix)
}

rotation = fit_linear_transform(mass_frac_data, mass_PCA$x)
stopifnot(all.equal(as.matrix(mass_frac_data)%*%rotation, mass_PCA$x))
mass_PCs = (as.matrix(mass_frac_data)%*%rotation)[,1:n_PCs]
PC_sources = (as.matrix(souspec_data)%*%rotation)[,1:n_PCs]

zmix = PCA_data[,'Zmix'] # including Zmix doesn't change very much...
Xpos = PCA_data[,'Xpos'] # including Xpos doesn't change very much...

cat('Compare predictability of CPV_sources from CPVs with that of PCA option...')
cat('regular PCA:')

zmix_PCs_to_PC_sources = get_explained_var_H2O(cbind(zmix,mass_PCs,Xpos), PC_sources, max_models=150, max_runtime_sec=8*60*60)
cat('zmix,mass_PCs --> PC_sources, R^2 = ', zmix_PCs_to_PC_sources) 
#zmix_PCs_to_all_dep = get_explained_var_H2O(cbind(zmix,mass_PCs,Xpos), all_dependants, max_models=150, max_runtime_sec=8*60*60)
#cat('zmix,mass_PCs --> all_dependants, R^2 = ', zmix_PCs_to_all_dep)
zmix_PCs_to_mass_frac = get_explained_var_H2O(cbind(zmix,mass_PCs,Xpos), mass_frac_data, max_models=150, max_runtime_sec=8*60*60)
cat('zmix,mass_PCs --> mass_frac_data, R^2 = ', zmix_PCs_to_mass_frac)

# cat('Learned CPVs:')
# cat('zmix,CPVs --> CPV_sources, R^2 = ', get_explained_var_H2O(cbind(zmix,CPVs,Xpos), CPV_sources))
# cat('zmix,CPVs --> all_dependants, R^2 = ', get_explained_var_H2O(cbind(zmix,CPVs,Xpos), all_dependants))
# cat('zmix,CPVs --> mass_frac_data, R^2 = ', get_explained_var_H2O(cbind(zmix,CPVs,Xpos), mass_frac_data))


