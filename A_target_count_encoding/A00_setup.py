###########################################################
##
## Modify paths here to point to your data sources, 
## and locations for temporary or output files
##
############################################################


# Directory for temporary or intermediate files
temp_path = '../data/2024_04_18'

# Parent data path
parent_path = '../data/2024_03_15'

# Predictors, copied from setup
predictor_features = ['NoEmp', 'CreateJob', 'LowDoc', 
       'DisbursementGross',  'new_business', 'urban_flag',
       'franchise_flag']


###########################################################
##
## Constants.  These do not require modification.
## These are values used across notebooks, or are
## long and placed here for convenience
##
###########################################################

#
# XGBoost parameters
#

xgb_n_estimators = 30

selected_lambda_k = 100

#
# Traditional NN parameters
#

nn_layer_sizes = [128, 64]
nn_dropout = 0.5
nn_batch_size = 32
nn_epochs = 20
nn_learning_rate = 0.0005
nn_naics_embed_size = 8

# Optimizer - you may want to change this based on your hardware
import tensorflow as tf
nn_optimizer = tf.keras.optimizers.legacy.Adam

#
# DGI parameters
#


# Optimizer - you may want to change this based on your hardware
gnn_optimizer = tf.keras.optimizers.legacy.Adam

# Unsupervised GNN - edge sampling in graph creation
gnn_unsup_sample = False
gnn_unsup_sample_n = 500
gnn_batch_size = 32

gnn_unsup_num_samples = [100]
gnn_unsup_layer_sizes = [8]
gnn_unsup_dropout = 0.2
gnn_unsup_activations = ['tanh']
gnn_unsup_epochs = 100
gnn_unsup_learning_rate = 0.01


