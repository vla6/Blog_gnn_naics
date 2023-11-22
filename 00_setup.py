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
temp_path = '../data/2023_10_07'


###########################################################
##
## Constants.  These do not require modification.
## These are values used across notebooks, or are
## long and placed here for convenience
##
###########################################################

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
