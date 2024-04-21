###########################################################
##
## Class definition for hierarchical blending
## https://dl.acm.org/doi/pdf/10.1145/507533.507538
##
############################################################

import pandas as pd
import numpy as np

default_lambda_k = 25
default_lambda_f = 20

class HierarchicalEncoder():
    """ Class to do hierarchical encoding """
    
    # Parameters
    lambda_k = default_lambda_k
    lambda_f = default_lambda_f
    
    # Set in fit
    stats_list = None
    overall_mean = None
    
    @staticmethod
    def blend_lambda(n, k=default_lambda_k, f=default_lambda_f):
        """Get blending ratio; Sigma-like blend"""
        return 1 / (1 + np.exp((k-n)/f))

    
    def stats_list_item(self, data):
        """Aggregate to get means for one category. 
          Inputs:
            data:  Pandas dataframe where the first column is the
                    category, and the second the target
          Value:
            Pandas dataframe with 4 columns:
              cat: The categories for encoding
              mean: The mean response
              count: The response count
              lambda:  The blending fraction given 'count'
          """
        
        data_agg = data \
            .set_axis(['cat', 'targ'], axis=1) \
            .reset_index(drop=True)\
            .groupby('cat') \
            ['targ'] \
            .agg(['mean', 'count']) \
            .reset_index()
        
        data_agg['lambda'] = data_agg['count'] \
            .apply(lambda x: HierarchicalEncoder.blend_lambda(x, self.lambda_k, self.lambda_f))
        return data_agg
    
    def get_stats_list(self):
        return self.stats_list
    
    def set_stats_list(self, X, y):
        """Sets a list of means, counts, and blend fractions for each level of the 
        hierarchy.  The last item corresponds to the overall mean
          Inputs:
            X: Category assignments in precedence order (Pandas DataFrame)
            y: Target values (Pandas Series)
        """
        major_col = list(X.columns)[0]
        map_master = X.drop_duplicates(major_col)
        
        # Get category values
        self.stats_list = [self.stats_list_item(pd.concat([X.iloc[:, i], y], axis=1)) \
                    for i in range(X.shape[1])]
        
        self.overall_mean = y.mean()
        
        #self.stats_list[-1] = pd.DataFrame({'cat':[None], 'mean':[y.mean()],
        #                                    'count':[y.count()], 'lambda':[1]})
        
    def fit(self, X, y):
        """ Fits the encoder. 
          Inputs:
            X: Category assignments in precedence order (Pandas DataFrame)
              Blending starts from the leftmost category and proceeds rightward
            y: Target values (Pandas Series)
        """
        # Set the blend ratios and levels
        self.set_stats_list(X, y)
        
    def transform(self, X):
        """ Transforms the input data, resulting in a target encoded value
        based on all the categories in X, in precedence order from left to right
          Input:
            X: Category assignments in precedence order (Pandas DataFrame)
                Blending starts from the leftmost category and proceeds rightward.
                Column names and order must match data sent to fit()
          Value: Pandas Series consisting of encoded value
        """
        
        # Cascading left join
        X_names = list(X.columns)
        join_df = pd.concat([X.iloc[:, [i]].merge(self.stats_list[i], left_on=X_names[i],
                                        right_on='cat', how='left') \
                                 .drop(columns='cat') \
                                 for i in range(len(X_names))],
                            axis = 1, keys=X_names)
        join_df.index = X.index
        
        # Fill NAs - missing values have 0 weight
        join_df.loc[:, (slice(None), 'lambda')] = \
            join_df.loc[:, (slice(None), 'lambda')].fillna(0)
        
        # Category keys
        levels_0_list =  list(dict.fromkeys(join_df.columns.get_level_values(0)))
        
        # Get the weighted means for each category separately
        blend_direct = pd.concat([join_df.xs(c, level=0, drop_level=True, axis=1)['mean'] * \
                          join_df.xs(c, level=0, drop_level=True, axis=1)['lambda'] \
                         for c in levels_0_list], axis=1, keys=levels_0_list)
        blend_direct['last'] = self.overall_mean
        
        # Get the (1-probabilities) from earlier terms
        blend_lambda_minus_1 = pd.concat([1 - join_df.xs(c, level=0, drop_level=True, axis=1)['lambda'] \
                         for c in levels_0_list], axis=1)
        blend_lambda_minus_1.insert(0, 'first_col', 1)
        blend_lambda_minus_1.columns = blend_direct.columns
        
        # Create cumulative prod
        blend_lambda_minus_1 = blend_lambda_minus_1.cumprod(axis=1)
        
        # Multiple direct blends by cumulative probability
        return blend_direct.mul(blend_lambda_minus_1).sum(axis=1)

    def fit_transform(self, X, y):
        """ Fits the encoder, and then transforms the input data
          Inputs:
            X: Category assignments in precedence order (Pandas DataFrame)
            y: Target values (Pandas Series)
          Value: Pandas Series consisting of encoded value
        """
        self.fit(X, y)
        return self.transform(X)
                                            
    
    def __init__(self, 
                 lambda_k = default_lambda_k,
                 lambda_f = default_lambda_f):
        """ Implement hierarchical encoding with sigmoidal blending.
          Attributes:
            lambda_k: Blending midpoint
            lambda_f:  Blending width
        """
        if lambda_k is None:
            self.lambda_k = default_lambda_k
        else:
            self.lambda_k = lambda_k
        if lambda_f is None:
            self.lambda_f = default_lambda_f
        else:
            self.lambda_f = lambda_f
            
        print(self.lambda_k)
        print(self.lambda_f)