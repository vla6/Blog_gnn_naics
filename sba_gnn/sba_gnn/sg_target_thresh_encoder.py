###########################################################
##
## Class definition to do simple thresholded target
## encoding, but use NA values for low-count cases
##
############################################################

import pandas as pd
import numpy as np

class TargetThreshEncoder():
    """ Class to do target+count encoding.  Does one feature at a time """
    
    encodings = None

        
    def fit(self, X, y, random_state = None):
        """Fits the training data.  Stores a dataframe of response estimates
        and code counts for each value.  
          Inputs:
            X: Category feature (Pandas Series)
            y: Target values (Pandas Series)
            random_state: Sets Numpy random seed for fold splitting in the
                data.
        """
        
        
        if X.name is None:
            X.rename('feature', inplace=True)
        
        # Create data frame with the X and y values
        df = pd.concat([X.rename('feature'), y], axis=1) \
            .set_axis(['feature', 'target'], axis=1) \
            .reset_index(drop=True)
        
        # Get counts and means
        df_agg = df.groupby('feature') \
            ['target'] \
            .agg(['mean', 'count']) \
            .reset_index() 
            
        # Threshold
        df_agg = df_agg[df_agg['count'] >= self.threshold]
        df_agg = df_agg[['feature', 'mean']] \
            .rename(columns={'mean':X.name}) 
        
        self.encodings = df_agg
        
    def transform(self, X):
        """ Apply the encodings to a dataset. 
          Inputs:
            X: Category feature (Pandas Series)
          Value:
            Pandas dataframe containing 2 columns representing
            the categorical value.
        """
        
        if X.index.name is not None:
            ind_nm = X.index.name
        else:
            ind_nm = 'index'

        # Map the encodings to the features
        enc_df = X.rename('feature') \
            .reset_index() \
            .merge(self.encodings, how='left', on= 'feature') \
            .drop(columns='feature') \
            .set_index(ind_nm)
        
        if self.fill_value is not None:
            enc_df = enc_df.fillna(self.fill_value)
        
        return enc_df
    
    def __init__(self, threshold = 100, fill_value = None):
        """ Target encoding, but threshold values below a certain point;
        missing will be NA (or, optionally another value)
          Attributes:
            threshold: counts below which we set encodings to NA
            fill_val: Optional value for filling NAs.  This is
              not recommended for XGBoost.  None means
              low volume, unknown, or missing are NA
        """
        self.threshold = threshold
        self.fill_value = fill_value
