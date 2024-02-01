###########################################################
##
## Helper functions for GNN data and models
## Used in multiple scripts
##
############################################################

import pandas as pd
import numpy as np

#
# Data processing (unsupervised model)s
#

def get_naics_index(naics_seq):
    return 'n_' + naics_seq

def get_naics_sector_index(naics_sec):
    return 'ns_' + naics_sec.astype('str')


def limit_data(edges_list, features_business, features_naics,
               dsets_list = ['train'],
              edge_types_retain = ['naics_sector']):
    """Limit cases input to GNN.  Takes in data for all cases
    and filters nodes and edges to the data sets (train, test, validation)
    specified.  Any combination of data sets can be outptut"""
    
    # Business features limit 
    features_business_lim = features_business[features_business['dset'].isin(dsets_list)] \
        .drop(columns='dset') 
    
    # Get NAICS associated with the businesses in this dataset 
    included_naics = features_business_lim['NAICS_orig'].drop_duplicates()
    naics_ind = get_naics_index(included_naics).rename('target')
    
    # Limit edges
    edges_lim = edges_list[edges_list['dset'].isin(dsets_list) |
                          edges_list['type'].isin(edge_types_retain)] \
        .drop(columns='dset') \
        .drop_duplicates() \
        .merge(naics_ind, on='target')
    
    # Limit NAICS
    features_naics_lim = features_naics.merge(naics_ind.to_frame().set_index('target'), 
                                              left_index=True, right_index=True)

    return (edges_lim, features_business_lim, features_naics_lim)


def graph_labels(label_data, G, business_node_type = 'LoanNr_ChkDgt'):
    """Get label data relevant for the Stellargraph, assuming fields
    used in this project
      Inputs: 
        label_data: Dataframe containing response features.  Indices
          expected to be business ID.
        G:  Stellargraph object containing business nodes.  
        business_node_type:  Node type for businesses in the Stellargraph
      Value: label_data values for businesses in the graph.  """
    
    selected_bus = pd.Series(G.nodes(node_type=business_node_type)) \
        .rename(business_node_type)
    
    label_data = label_data.copy()
    label_data.index.rename(business_node_type, inplace=True)
    graph_labels = label_data.reset_index() \
        .merge(selected_bus, on=business_node_type)
    
    return graph_labels