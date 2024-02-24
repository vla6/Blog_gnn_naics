###########################################################
##
## Modify paths here to point to your data sources, 
## and locations for temporary or output files
##
############################################################

# Input data
# Download from https://www.kaggle.com/datasets/mirbektoktogaraev/should-this-loan-be-approved-or-denied/
# The path below should point to this data on your system.

input_path = '../kaggle_input/SBAnational.csv'

# Directory for temporary or intermediate files
temp_path = '../data/2024_02_23'


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
# GNN parameters
#

# Training parameters
gnn_batch_size = 32
gnn_learning_rate = 0.002
gnn_dropout = 0.5
gnn_epochs = 20

gnn_num_samples = [10, 20]
gnn_layer_sizes = [64, 64]
gnn_activations = ['tanh', 'tanh']


# Optimizer - you may want to change this based on your hardware
gnn_optimizer = tf.keras.optimizers.legacy.Adam

# Unsupervised GNN - edge sampling in graph creation
gnn_unsup_sample = False
gnn_unsup_sample_n = 500

gnn_unsup_num_samples = [100]
gnn_unsup_layer_sizes = [8]
gnn_unsup_dropout = 0.2
gnn_unsup_activations = ['tanh']
gnn_unsup_epochs = 100
gnn_unsup_learning_rate = 0.01


#
# Features to include in models
# Use a limited list to reduce runtime
# 

#predictor_features = ['NoEmp', 'CreateJob', 'RetainedJob', 'LowDoc', 'RevLineCr',
#       'DisbursementGross', 'GrAppv', 'guaranteed_fract', 'new_business', 'urban_flag',
#       'franchise_flag']

predictor_features = ['NoEmp', 'CreateJob', 'LowDoc', 
       'DisbursementGross',  'new_business', 'urban_flag',
       'franchise_flag']



#
# Data types information for 01_data_import
#

input_dtypes = {'LoanNr_ChkDgt':'str',
                'Name':'str',
                'City':'str',
                'State':'str',
                'Zip':'str',
                'Bank':'str',
                'BankState':'str',
                'NAICS':'str',
                'ApprovalDate':'str',
                'ApprovalFY':'str',
                'Term':'int',
                'NoEmp':'int',
                'NewExist':'str',
                'CreateJob':'int',
                'RetainedJob':'int',
                'FranchiseCode':'str',
                'UrbanRural':'str',
                'RevLineCr':'str',
                'LowDoc':'str',
                'ChgOffDate':'str',
                'DisbursementDate':'str',
                'MIS_Status':'str'}

# Date fields to be converted when data read
input_dates = ['ApprovalDate', 'ChgOffDate', 'DisbursementDate']


# Function and dictionary for conversion of currency fields

import pandas

def currency_converter(str):
    return float(str.replace('$', '').replace(',', ''))

input_converters = {'SBA_Appv':currency_converter,
                  'GrAppv':currency_converter,
                  'ChgOffPrinGr':currency_converter,
                  'BalanceGross':currency_converter,
                  'DisbursementGross':currency_converter}
