##
## Plotting and metric functions for use across notebooks
##

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import textwrap

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, \
    average_precision_score, roc_auc_score

from sklearn.preprocessing import TargetEncoder

#
# Plot setups, default parameters
#


def plot_defaults():
    """ Set default plot parameters"""
    plt.style.use('seaborn-v0_8-white')
    mpl.rcParams.update({'font.size': 16})
    mpl.rcParams.update({'axes.titlesize': 18})
    
#
# Basic bar plot
#

def plot_basic_bar(data, y, 
                   label = None,
                   n_bars = 10,
                   figsize = None,
                   ylabel = None,
                   title=None,
                   do_sort = False,
                   zero_line = True,
                   width = 0.9):
    """ Create a basic bar plot for a Pandas dataframe."""
    
    if do_sort:
        data = data.copy().sort_values(y, ascending=False)
        
    if label != None:
        data = data.copy().set_index(label)
        
    if ylabel == None:
        ylabel = data[y].name
        
    # Set figsize if not explicit
    if figsize == None:
        figsize = (4, n_bars/3.3)
        
    fig, ax = plt.subplots()
    
    data.head(n_bars)[[y]] \
        .plot(kind='barh', legend=None, figsize=figsize, ax=ax, width=width)
    
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_ylabel(None)
    ax.set_xlabel(ylabel)
    
    if (zero_line):
        plt.axvline(x=0,color='gray')

    
    return fig

#
# NAICS variation plot
#

def wald_interval_sd(r, c):
    """Calculate the Wald interval for the target rate in a group
      Inputs:
        r: Mean response (rate)
        c:  Number of observations
    """
    #print(c)
    #print(r)
    return np.sqrt(r * (1-r)) / np.sqrt(c)

def naics_variance_data(data, naics_feat='NAICS',
                        naics_sector_feat = 'NAICS_sector',
                        naics_sector_desc_feat = 'NAICS_sector_desc',
                       target_col = 'target',
                       id_col = 'LoanNr_ChkDgt'):
    """ Aggregate data by NAICS """
    naics_sum_1 = data \
        .groupby(naics_feat) \
        .agg({target_col:'mean', id_col:'count', naics_sector_feat:'first'}) \
        .rename(columns={id_col:'count_naics', target_col:'target_naics'}) \
        .reset_index()
    
    naics_sum_2 = data\
        .groupby(naics_sector_feat) \
        .agg({id_col:'count', target_col:'mean', naics_sector_desc_feat:'first'})\
        .rename(columns={id_col:'count_cat', target_col:'target_cat'}) \
        .reset_index() \
        .sort_values('count_cat', ascending=False)
    
    naics_sum = naics_sum_1.merge(naics_sum_2, how='inner', on=naics_sector_feat) \
        .sort_values(['count_cat', 'count_naics'], ascending=[False, False])
 
    naics_sum['target_naics_sd'] = naics_sum.apply(lambda x: wald_interval_sd(x.target_naics, x.count_naics),
                                            axis=1)
    naics_sum['target_cat_sd'] = naics_sum.apply(lambda x: wald_interval_sd(x.target_cat,x.count_cat),
                                                axis=1)
    return naics_sum

def naics_variance_plot(data_agg, naics_feat = 'NAICS',
                        naics_sector_feat = 'NAICS_sector',
                        naics_sector_desc_feat = 'NAICS_sector_desc',
                        naics_sector_sort = True,
                        x='target_naics',
                        xerr = 'target_naics_sd',
                        x_sector = 'target_cat',
                        x_sector_err = 'target_cat_sd',
                        x_sector_count = 'count_cat',
                        xlabel = 'target rate',
                        num_sectors = None,
                        naics_int = False,
                        fig_width = 7, fig_height_sec = 1):
    
    data_agg = data_agg.copy()
    if naics_sector_sort:
        data_agg.sort_values([x_sector_count, naics_feat], ascending=[False, True], inplace=True)
    if naics_int:
        data_agg[naics_feat] = data_agg[naics_feat].astype('str').astype('int')    
    
    # Plot a subset of sectors if applicable
    if num_sectors is None:
        num_sectors = data_agg[naics_sector_feat].drop_duplicates().count()
    
    # Create the plots
    plt.close()
    fig, ax = plt.subplots(num_sectors, 1, sharex=True, figsize=(fig_width, fig_height_sec * num_sectors))

    i = 0
    for name, group in data_agg.groupby(naics_sector_feat, sort=False):
        group.plot(y=naics_feat, x=x, xerr=xerr, ax=ax[i], kind='scatter', marker='|')
    
        this_x1 = (group[x_sector] -  group[x_sector_err]).iloc[0]
        this_width = 2*group[x_sector_err].iloc[0]
        this_desc = str(name) + ': ' + group[naics_sector_desc_feat].iloc[0]
        this_desc = "\n".join(textwrap.wrap(this_desc, 30))
    
        ax[i].set_ylabel(None)
        ax[i].xaxis.set_major_formatter(ticker.PercentFormatter(1))
        ax[i].set_title(this_desc, size=12, x = 1.01, y=0.5, ha='left')
        
        # Adjust labels if axes are categorical
        if not naics_int:
            plt.draw()
            y_label = [item.get_text() for item in ax[i].yaxis.get_ticklabels()]
            new_lbl = [str(x) if i in [int(np.floor(len(y_label)/4)), len(y_label)-1] \
                        else '' for i, x in enumerate(y_label) ]
            ax[i].yaxis.set_ticks(ax[i].get_yticks())
            ax[i].yaxis.set_ticklabels(new_lbl)
            ax[i].set_yticklabels(new_lbl)
            ax[i].margins(y=0.08)
        ax[i].tick_params(axis='both', which='major', labelsize=12)
            
        #plt.margins(x=0)
        this_ylim= ax[i].get_ylim()
        ax[i].add_patch(patches.Rectangle(xy=(this_x1, this_ylim[0]), 
                                          width=this_width, height = this_ylim[1] - this_ylim[0],
                                          color='gray', fill=True, alpha=0.5))
        
        i = i + 1
        if i >= num_sectors:
            break   

    ax[i-1].set_xlabel(xlabel)
    ax[int(np.floor(num_sectors/2))].set_ylabel('NAICS')
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.03)
    
    return fig

#
# Threshold tuning curve 
#

def get_f1_frame(actual, pred_prob, num_pts = 51):
    """Given actual responses and model probability predictions, get f1 scores over a
    range of thresholds
      Inputs:
        actual: Pandas series containing actual target values (0/1)
        pred_prob:  Pandas series containing probability predictions
        num_pts:  Number of threshold points to examine
      Value: Dataframe consistig of a threshold and f1 score"""
    thresh_ser = pd.Series(np.linspace(0, 1, num_pts))
    return pd.DataFrame({'thresh': thresh_ser,
                         'f1': thresh_ser.apply(lambda x: f1_score(actual, 
                                                                   get_binary_predictions(pred_prob, x)))})

def get_binary_predictions(pred_prob, thresh):
    """Given probability predictions and a decision thresold, return binary predictions
      Inputs:
        pred_prob: Pandas series containing model probability predictions
        thresh:  Threshold above which we predict a 1 outcome
      Value: Series containing binary predictions
    """
    return pd.Series(np.where(pred_prob > thresh, 1, 0),
                    index=pred_prob.index).astype(int)

#
# Standard metrics
#

def do_metric(metric, actual, predict_bin, predict_prob):
    """ Get a sklearn metric, selecting the proper inputs and returning
    np.nan on error
    Inputs:
        metric: sklearn metric. Expected to either be average_precision_score or
          roc_auc_score, or to take the actual and binary predictions as inputs.
        actual: Pandas series containing actual outcomes
        predict_bin: Pandas series containing binary predictions
        predict_prob: Pandas series containing probability predictions
      Value: Metric for the data (float)
    """
    try:
        if metric in [average_precision_score, roc_auc_score]:
            return metric(actual, predict_prob)
        else:
            return metric(actual, predict_bin)
    except:
        return np.nan
    

def dset_metrics(actual, predict_bin = None, predict_prob = None,
                 metrics_list = [accuracy_score, f1_score, precision_score, recall_score, 
                                average_precision_score, roc_auc_score]):
    """ Return a Series containing standard metrics for the binary classification model.
    Inputs:
        actual: Pandas series containing actual outcomes
        predict_bin: Pandas series containing binary predictions.  Optional (but either
          predict_bin or predict_prob will be needed to generate metrics)
        predict_prob: Pandas series containing probability predictions.  Optional (but 
          either predict_bin or predict_prob will be needed to generate metrics)
        metrics_list: List of classification metrics to return
      Value: Series containing metrics data for the inputs 
    """
    return pd.Series([do_metric(m, actual, predict_bin, predict_prob) \
                      for m in metrics_list], index=[m.__name__ for m in metrics_list]) 

#
# Mean encoding within groups
# Using Scikit Learn
#

# Per group function
def mean_enc_grp(data, feature = 'NAICS', feature_out = 'menc_grp_NAICS', random_state=3453,
                cv = 5):
    """ Mean encodes within groups.  Expects input grouped data, with 'dset' feature = 'train
    for the training slice.  For one group, an encoder is created and then fit to the training
    data.  All rows are transformed.  Groups with too low volume in the test and/or train data
    are set to NA"""
    
    naics_grp_encoder = TargetEncoder(target_type='binary', random_state=random_state, cv = cv)
    naics_grp_encoder.set_output(transform='pandas')
    
    # We need a baseline number of rows to encode
    if len(data[data['dset'] == 'train']) < cv:
        return pd.DataFrame({feature_out:[np.nan]*len(data)}, index=data.index)
    
    # Some categories may be too low volume to encode; return NAs then
    try:
        train_out = naics_grp_encoder.fit_transform(data[data['dset'] == 'train'][[feature]], 
                      data[data['dset'] == 'train']['target'])
        other_out= naics_grp_encoder.transform(data[data['dset'] != 'train'][[feature]])
        out_data = pd.concat([train_out, other_out], axis=0).sort_index()
        out_data.columns = [feature_out]
    except:
        out_data = pd.DataFrame({feature_out:[np.nan]*len(data)}, index=data.index)
    return out_data

#
# Summary level stats
#

def naics_grp_stats(data, group_col = 'NAICS_sector'):
    """Count loans and NAICS codes within groups"""
    naics_agg = data.groupby(group_col)['target'] \
        .agg(['count', 'mean']) \
        .set_axis(['loan_count', 'target'], axis=1) \
        .reset_index()

    naics_agg['low_vol'] = np.where(naics_agg['loan_count'] <=50, 1, 0)
    grp_agg = naics_agg \
        [['loan_count', 'target', 'low_vol']] \
        .agg(['count', 'mean', 'median', 'min', 'max', 'sum'])
    
    naics_code_agg = data.drop_duplicates('NAICS') \
        .groupby(group_col)['target'] \
        .agg(['count']) \
        .set_axis(['naics_count'], axis=1) \
        .reset_index()
    
    naics_code_agg['single_naics'] =  np.where(naics_code_agg['naics_count'] == 1, 1, 0)
    
    naics_code_agg_grp = naics_code_agg \
        [['naics_count', 'single_naics']] \
        .agg(['count', 'mean', 'median', 'min', 'max', 'sum']) 
    
    grp_agg = pd.concat([grp_agg, naics_code_agg_grp], axis=1)
    grp_agg = grp_agg.transpose() \
        .rename(columns={'count':'count_grp'})
    return grp_agg