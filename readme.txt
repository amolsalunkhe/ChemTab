PAPER-1-IMAGES.pptx contains all the colored images

All the other images come from the notebook ("PCDNN_PAPER1.ipynb"). 

The data needed is "NewData_flames_data_with_L1_L2_errors_CH4-AIR_with_trimming.txt"

In the second cell of the Notebook change the following to wherever you place the data
df = pd.read_csv('C:\\Users\\amol\\Documents\\PHD\\DISSERTATION\\NewData_flames_data_with_L1_L2_errors_CH4-AIR_with_trimming.txt')

Experiment I (Zmix, Cpv)
 A. Training using only the data from the flames used by the framework
    i>  Gaussian Process
    ii> DNN
 B. Training 50% random/scrambled data 
    i>  Gaussian Process
    ii> DNN

Experiment II (Zmix, 4PCAs)
 A. Training using only the data from the flames used by the framework
    i>  Gaussian Process
    ii> DNN
 B. Training 50% random/scrambled data 
    i>  Gaussian Process
    ii> DNN

Experiment III (All Species)
A. Training 50% random/scrambled data 
    i> PC - DNN
    ii> Unconstrained - DNN

Each of the experiment has a Error Residual Plot and the DNNs also may have a Training Loss Plot

Naming convention for "Error Residual Plot" is as follows:
   Model Type: gp/dnn/pcdnn
   Error Observation: uncorrelated_errors/correlated_errors	   
   Input: zmix_cpv/zmix_pca/(all species is nothing)

e.g. gpuncorrelated_errors_zmix_pca.png


Tip: Run the GP Experiments the last it take a lot of time to run
 	