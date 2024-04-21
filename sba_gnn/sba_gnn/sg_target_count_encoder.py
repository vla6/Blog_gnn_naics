###########################################################
##
## Class definition to do target-count encoding.  
## Combines
##
############################################################

import pandas as pd
import numpy as np

import sba_gnn.sba_gnn.sg_blender as sg_blender
from sba_gnn.sba_gnn.sg_blender import HierarchicalEncoder

default_lambda_k = sg_blender.default_lambda_k
default_lambda_f = sg_blender.default_lambda_f

class TargetCountEncoder():
    """ Class to do target+count encoding.  Does one feature at a time """
    
    hier_enc = None
    counts = None
    
    def fit(self, X, y, random_state = None):
        """Fits the training data.  Stores a dataframe of response estimates
        and code counts for each value.  
          Inputs:
            X: Category feature (Pandas Series)
            y: Target values (Pandas Series)
            random_state: Sets Numpy random seed for fold splitting in the
                data.
        """
        
        self.hier_enc = HierarchicalEncoder(lambda_k = self.lambda_k,
                                            lambda_f = self.lambda_f)
        
        self.hier_enc.fit(X.to_frame(), y)
        
        self.counts = self.hier_enc.get_stats_list()[0][['cat', 'count']] \
            .rename(columns={'cat':X.name})
        
        if self.count_threshold is not None:
            self.counts['count'] = self.counts['count'] \
                .where(self.counts['count'] <= self.count_threshold,
                       self.count_threshold)
        
    def transform(self, X):
        """ Apply the encodings to a dataset. 
          Inputs:
            X: Category feature (Pandas Series)
          Value:
            Pandas dataframe containing 2 columns representing
            the categorical value.
        """
        
        # Get the response part from the existing encoder
        response_ser = self.hier_enc.transform(X.to_frame()) \
            .rename('response')
        
        # Get the counts part from a match
        start_index = X.index
        count_df = X.reset_index(drop=True) \
            .to_frame() \
            .merge(self.counts, how='left', on= X.name) \
            .drop(columns=X.name) \
            .sort_index()
        count_df.index = start_index
        count_df['count']= count_df['count'].fillna(0)
        
        enc_df = pd.concat([response_ser, count_df], axis = 1)
        
        return enc_df
    
    def __init__(self, 
                 lambda_k = default_lambda_k,
                 lambda_f = default_lambda_f,
                 count_threshold = True):
        """ Implement hierarchical encoding with sigmoidal blending.
          Attributes:
            lambda_k: Blending midpoint
            lambda_f:  Blending width
            count_threshold:  If True, set to lambda_k + 3*lambda_f.
              If None or False, do not threshold.  If an integer, threshold 
              below this level.
        """
        if lambda_k is None:
            self.lambda_k = default_lambda_k
        else:
            self.lambda_k = lambda_k
        if lambda_f is None:
            self.lambda_f = default_lambda_f
        else:
            self.lambda_f = lambda_f
        
        if isinstance(count_threshold, bool):
            if count_threshold:
                self.count_threshold = self.lambda_k + 3*self.lambda_f
            else:
                count_threshold = None
        else:
            self.count_threshold = count_threshold