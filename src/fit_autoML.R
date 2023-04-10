library(dplyr)
library(tidyr)
source('~/.RProfile')

library(h2o)
h2o.init()

# NOTE: it uses validation data for r^2! Wahoo! (CV R^2 doesn't work for some reason...)
get_explained_var_H2O = function(X_df, Y_df, max_models = 30, max_runtime_secs=60) {
    # Run AutoML for estimating explained variance
    df = cbind(X_df, Y_df)
    aml_R2 = function(Y_col) { # coupled with outer function
        train_idx = sample(nrow(df), as.integer(0.8*nrow(df)))
        train_df = df[train_idx,]
        val_df = df[-train_idx,]
        aml <- h2o.automl(x=colnames(X_df), y=Y_col, training_frame = as.h2o(train_df), validation_frame=as.h2o(val_df),
                              max_models = max_models, max_runtime_secs=max_runtime_secs, nfolds=5)
        leader = h2o.get_best_model(aml)
        return(h2o.r2(leader, valid=T))
    }
            
    library(tidyr)
    library(purrr)
    colnames(Y_df) %>% purrr::map_dbl(aml_R2) %>% mean(na.rm=T) # why does R^2 get NaNs??
}

PCA_data = read.csv('models/best_models/PCDNNV2Model-0.9055090522558945/PCA_data.csv')
mass_frac_data = PCA_data %>% select(starts_with('Yi'))
souspec_data = PCA_data %>% select(starts_with('souspec'))#, souener)
#mass_PCs = PCA_data %>% select(matches('Pure_PCA_[0-9]+'))
CPVs = PCA_data %>% select(matches('PCDNNV2_PCA_[0-9]+'))
CPV_sources = PCA_data %>% select(matches('PCDNNV2_PCA_source_[0-9]+'))
souspec_data[,53] = 0

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
    for (j in 1:ncol(Y_df)) {
        model = lm(Y_df[,j]~.-1, data=X_df) # TODO: verify that it is ok to ignore bias here?
        rotation_matrix = cbind(rotation_matrix, coef(model))
    } 
    colnames(rotation_matrix) = colnames(Y_df)
            
    # print R^2 of entire rotation matrix
    MSE = apply((as.matrix(X_df)%*%rotation_matrix-Y_df)**2, -1, mean)
    VAR = apply(Y_df, -1, var)
    R2 = 1-sum(MSE/VAR)
    cat('R^2 of linear transform fit: ', R2)
                        
    return(rotation_matrix)
}

rotation = fit_linear_transform(mass_frac_data, mass_PCA$x)
stopifnot(all.equal(as.matrix(mass_frac_data)%*%rotation, mass_PCA$x))
mass_PCs = (as.matrix(mass_frac_data)%*%rotation)[,1:n_PCs]
PC_sources = (as.matrix(souspec_data)%*%rotation)[,1:n_PCs]

zmix = PCA_data[,'Zmix'] # including Zmix doesn't change very much...
Xpos = PCA_data[,'Xpos'] # including Xpos doesn't change very much...

cat('Compare predictability of CPV_sources from CPVs with that of PCA option...')
cat('Learned CPVs:')
cat('zmix,CPVs --> CPV_sources, R^2 = ', get_explained_var_H2O(cbind(zmix,CPVs,Xpos), CPV_sources))
cat('zmix,CPVs --> all_dependants, R^2 = ', get_explained_var_H2O(cbind(zmix,CPVs,Xpos), all_dependants))
cat('zmix,CPVs --> mass_frac_data, R^2 = ', get_explained_var_H2O(cbind(zmix,CPVs,Xpos), mass_frac_data))

cat('regular PCA:')
cat('zmix,mass_PCs --> PC_sources, R^2 = ', get_explained_var_H2O(cbind(zmix,mass_PCs,Xpos), PC_sources))
cat('zmix,mass_PCs --> all_dependants, R^2 = ', get_explained_var_H2O(cbind(zmix,mass_PCs,Xpos), all_dependants))
cat('zmix,mass_PCs --> mass_frac_data, R^2 = ', get_explained_var_H2O(cbind(zmix,mass_PCs,Xpos), mass_frac_data))

