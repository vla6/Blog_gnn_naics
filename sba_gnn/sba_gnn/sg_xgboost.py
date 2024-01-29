#
# Functions related to XGBoost modeling
#

import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
import matplotlib.pyplot as plt

import sklearn as sk
from sklearn.model_selection import RandomizedSearchCV

#
# Hyperparameter tuning
#

hyperparam_serarch = {'max_depth': [6, 8, 19],
                      'min_child_weight': [1, 10, 50],
                      'gamma': [0, 0.5, 1, 2],
                      'subsample': [0.5, 0.8, 1.0],
                      'learning_rate': [0.1, 0.3],
                      'reg_alpha': [0, 0.001, 0.01, 0.1],
                      'reg_lambda': [0.001, 0.01, 0.1, 0.5]
                     }


def hyperparameter_tune(X, y, 
                        n_estimators = 30, 
                        n_iter = 25,
                        space = hyperparam_serarch,
                        random_state = None,
                       pos_wt = None, pos_wt_levels = 4):
    
    xgb_model = xgb.XGBClassifier(objective="binary:logistic",
                             n_estimators = n_estimators)
    
    if pos_wt is not None:
        space['scale_pos_weight']= list(np.linspace(1, pos_wt, pos_wt_levels))
    
    # Set up a cross validation without the final fit; don't use early stopping yet
    random_search = RandomizedSearchCV(xgb_model, space,
                                   n_iter=20, 
                                   n_jobs=-1,
                                   random_state=random_state,
                                   verbose = 0,
                                   refit = False)
    
    # Do the cross validation parameter search
    rs_fit = random_search.fit(X, y);

    return random_search.best_params_