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
# Plotting labels and parameters
#

import pandas as pd

# Dictionaries for plotting, also ideal order
feature_order = ['menc', 'mhier', 'menc_all', 'tc', 'tt']
model_label_dict = {'menc': 'Target Encoding (NAICS only)',
                    'mhier': 'Hierarchical Blending',
                    'menc_all': 'Target Encoding (all)',
                    'tc': 'Target+Count Encoding',
                    'tt': 'Target-Thresh Encoding'}

model_label_colors = {'menc':'darkgray',
                      'mhier':'darkslateblue',
                      'menc_all': 'mediumaquamarine',
                      'tc':'darkorange',
                      'tt':'violet'}
model_label_styles = {'menc':'o-',
                      'mhier':'v-',
                      'menc_all': 'd-',
                      'tc':'+-',
                      'tt':'P-'}

model_label_type = pd.CategoricalDtype(categories = [model_label_dict[k] 
                                                     for k in feature_order], ordered=True)

# Funcion for applying labels
def label_models(data, feature  = 'model'):
    ser  = data[feature].apply(lambda x: model_label_dict[x]) \
        .astype(model_label_type)
    return ser
