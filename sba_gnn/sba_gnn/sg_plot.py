##
## Plotting functions for use across notebooks
##

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import textwrap

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

def naics_variance_data(data, naics_feat='NAICS',
                        naics_sector_feat = 'NAICS_sector',
                        naics_sector_desc_feat = 'NAICS_sector_desc',
                       target_col = 'target',
                       id_col = 'LoanNr_ChkDgt'):
    """ Aggregate data by NAICS """
    naics_sum_1 = data \
        .groupby(naics_feat) \
        .agg({target_col:'mean', id_col:'count', naics_sector_feat:'first'}) \
        .rename(columns={id_col:'count', target_col:'target'}) \
        .reset_index()
    
    naics_sum_2 = data\
        .groupby(naics_sector_feat) \
        .agg({id_col:'count', target_col:'mean', naics_sector_desc_feat:'first'})\
        .rename(columns={id_col:'count_cat', target_col:'target_cat'}) \
        .reset_index() \
        .sort_values('count_cat', ascending=False)
    
    naics_sum = naics_sum_1.merge(naics_sum_2, how='inner', on=naics_sector_feat) \
        .sort_values(['count_cat', 'count'], ascending=[False, False])
    
    naics_sum['target_sd'] = np.sqrt(naics_sum['count']* naics_sum['target']) / naics_sum['count']
    naics_sum['target_cat_sd'] = np.sqrt(naics_sum['count_cat'] * naics_sum['target_cat']) / \
        naics_sum['count_cat']
    
    return naics_sum

def naics_variance_plot(data_agg, naics_feat = 'NAICS',
                        naics_sector_feat = 'NAICS_sector',
                        naics_sector_desc_feat = 'NAICS_sector_desc',
                        naics_sector_sort = True,
                        x='target',
                        xerr = 'target_sd',
                        x_sector = 'target_cat',
                        x_sector_err = 'target_cat_sd',
                        num_sectors = None,
                        naics_int = False,
                        fig_width = 7, fig_height_sec = 1):
    
    data_agg = data_agg.copy()
    if naics_sector_sort:
        data_agg.sort_values(['count_cat', naics_feat], ascending=[False, True], inplace=True)
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

    ax[i-1].set_xlabel('target rate')
    ax[int(np.floor(num_sectors/2))].set_ylabel('NAICS')
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.03)
    
    return fig