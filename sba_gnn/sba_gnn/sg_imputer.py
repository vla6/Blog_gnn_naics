###########################################################
##
## Class definition for imputer to transform 
## tabular data to a format suitable for GNN.  
## No
##
############################################################

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

class GNNImputer():
    """ Class that contains several sklearn imputers to convert 
    tabular small business administration data to a format that
    is suitable for the GNN.  Specifically, missing values are
    median filled, (probably) continuous features are quantile scaled, 
    and remaining low-cardinality features are MinMax scaled.  """
    
    def median_imputer_trans_to_pandas(self, mat, index=None):
        """ Takes in matrix-format data output from the median imputer, and 
        converts to pandas.  Optionally add an index"""
        return pd.DataFrame(mat, columns= self.median_imputer.get_feature_names_out(),
                           index = index)
    
    def median_imputer_create_fit(self, data, transform=True):
        """ Create and fit the median imputer.  Optionally also transform the
        input data as it will be used for scaling"""
        self.median_imputer = SimpleImputer(missing_values=np.nan, 
                                            strategy='median',
                                            add_indicator=True)  
        if transform:
            data_trans = \
                self.median_imputer_trans_to_pandas(self.median_imputer \
                                               .fit_transform(data[self.features_in]),
                                                   index = data.index)
            return data_trans
        else:
            self.median_imputer.fit(data[self.features_in])
            return None
        
    def median_imputer_transform(self, data):
        """ Transform data using the median imputer, returning
        Pandas data"""
        return self.median_imputer_trans_to_pandas(self.median_imputer \
                                                   .transform(data[self.median_imputer.feature_names_in_]),
                                                  index = data.index)
    
    def set_scaled_features(self, data):
        """ Get a list of features that are probably binary or 
        categorical from the training data.  This is based on level
        counts.  If the data set is over a threshold number of
        rows, a sample of data is used"""
        
        # Get level counts
        sample_n = np.min([data.shape[0], self.num_levels_scale_sample])
        levels_df = pd.concat([pd.DataFrame([data[c].sample(sample_n) \
                                            .value_counts().count()])  \
                              for c in self.features_in],
                             keys=self.features_in)
        
        # Assume features at or below the threshold are not quantile scaled, instead use minmax
        self.features_minmax_scale = list(levels_df[levels_df[0] <= self.num_levels_scale] \
                                  .index.get_level_values(0))
        self.features_quantile_scale = [c for c in self.features_in if c not in self.features_minmax_scale]
    
    def scaler_trans_append(self, data_in, data_trans, scaler):
        """Post-processing of data output by a scaler, to ensure we keep all
        fields, not just those scaled.  """
        
        trans_df = pd.DataFrame(data_trans,
                                columns=scaler.feature_names_in_)
        trans_df.index = data_in.index
        
        # Get columns not affected by the transformation
        fix_cols = [c for c in data_in.columns if c not in trans_df.columns]
        
        # Return recombined data with same column order
        return pd.concat([data_in[fix_cols], trans_df], axis=1)[data_in.columns]
        
    def quantile_scaler_create_fit(self, data_fill, transform = True):
        """Takes in median-filled data and creates/fits a quantile scaler,
        if applicable.  If there are no continuous predictor, no scaler is fit.
        Requires set_scaled_features() to have been called """
        
        # Ignore quantile scaling if no eligible features
        if (len(self.features_quantile_scale) == 0) & (transform):
            return data_fill
        elif (len(self.features_quantile_scale) == 0):
            return None
        
        # Create and fit the scaler
        self.quantile_scaler = QuantileTransformer(n_quantiles = self.quantile_levels)
        
        if transform:
            trans_data= self.quantile_scaler.fit_transform(data_fill[self.features_quantile_scale])
            return self.scaler_trans_append(data_fill, trans_data,
                                           scaler = self.quantile_scaler)
        else:
            self.quantile_scaler.fit(data_fill[self.features_quantile_scale])
            return None
    
    def quantile_scaler_transform(self, data):
        """Quantile scale data, using a fitted quantile scaler"""
        scaled_data= self.quantile_scaler.transform(data[self.features_quantile_scale])
        return self.scaler_trans_append(data, scaled_data, scaler=self.quantile_scaler)
    
    def minmax_scaler_create_fit(self, data_fill, transform = True):
        """Takes in median-filled data and creates/fits a minmax scaler,
        if applicable.  If there are no low-cardinality predictors, no scaler 
        is fit. Requires set_scaled_features() to have been called """
        
        # Ignore quantile scaling if no eligible features
        if (len(self.features_minmax_scale) == 0) & (transform):
            return data_fill
        elif (len(self.features_minmax_scale) == 0):
            return None
        
        # Create and fit the scaler
        self.minmax_scaler = MinMaxScaler(clip=True)
        
        if transform:
            trans_data= self.minmax_scaler.fit_transform(data_fill[self.features_minmax_scale])
            return self.scaler_trans_append(data_fill, trans_data,
                                           scaler=self.minmax_scaler)
        else:
            self.minmax_scaler.fit(data_fill[self.features_minmax_scale])
            return None
        
    def minmax_scaler_transform(self, data):
        """Quantile scale data, using a fitted quantile scaler"""
        scaled_data= self.minmax_scaler.transform(data[self.features_minmax_scale])
        return self.scaler_trans_append(data, scaled_data, scaler=self.minmax_scaler)

    def fit_transform(self, data, transform = True):
        """Fit the imputer/scaler, and return transformed training data.  
        Median fills nulls and creates null indicator features.  
        Quantile transforms many-level fields, and min/max scales all 
        fields.  Input fields must be numeric and set during initialization."""
        
        # Create/fit median imputer for missing data, transform training data
        trans_data1 = self.median_imputer_create_fit(data, transform=True)
        
        # Figure out which features to scale and not scale
        self.set_scaled_features(data)
        
        # Crete/fit the quantile scaler
        trans_data2 = self.quantile_scaler_create_fit(trans_data1, transform = transform)
        if not transform:
            trans_data2 = trans_data1
        
        # Create / fit the minmax scaler
        trans_data3 = self.minmax_scaler_create_fit(trans_data2, transform = transform)
        
        # Save the features after transform
        self.features_out = list(trans_data3.columns)
        
        return trans_data3

        
    def fit(self, data):
        """Fit the scalers, do not return transformed trainin gdata"""
        self.fit_transform(data, transform=False)
        
    def transform(self, data):
        """Transform dataset containing features, which will have null
        values median filled, and scaled. """
        data_1 = self.median_imputer_transform(data)
        data_2 = self.quantile_scaler_transform(data_1)
        return self.quantile_scaler_transform(data_2)
    
    def __init__(self, features = None, 
                 num_levels_scale = 5, num_levels_scale_sample = 100000,
                quantile_levels = 1000):
        """ Instantiates the custom scaler.  
          Inputs:
            features:  List of input features affected by the transformations.
              Other features in data passed to fit/transform functions
              will be ignored.  If None, all training data features are used 
              (and all data sets passed to fit or transform must have the
              same features)
            num_levels_scale: If a feature contains this or fewer unique values,
              I use minmax scaling.  Above the threshold, quantile scaling is used.
            num_levels_scale_sample:  For efficiency on large data sets, a sample
              of cases is used to determine the number of levels per feature
              to determine whether the feature should be quantile or minmax scaled.
            quantile_levels: Number of quantiles to use for the quantile scaling
              
        """
        self.features_in = features 
        self.num_levels_scale = num_levels_scale
        self.num_levels_scale_sample = num_levels_scale_sample
        self.quantile_levels = quantile_levels
        
        # During fit, I select some features for quantile scaling, others for
        # simple minmax scaling, based on level count, i.e. which are likely to
        # be categorical or binary.  
        features_quantile_scale = None
        features_minmax_scale = None
        
        # Output features may be more than input, set during fit()
        self.features_out = None
        
        # scikit-learn imputers/scalers to be created during fit() 
        median_imputer = None
        quantile_scaler = None
        minmax_scaler = None