###########################################################
##
## Helper functions for GNN data and models
## Used in multiple scripts
##
############################################################

import pandas as pd
import numpy as np

#
# Data processing
#

def limit_data(edges_list, features_business, features_naics,
               dsets_list = ['train'],):
    """Limit cases input to GNN.  Takes in data for all cases
    and filters nodes and edges to the data sets (train, test, validation)
    specified.  Any combination of data sets can be outptut"""
    
    # Limit edges
    edges_lim = edges_list[edges_list['dset'].isin(dsets_list)] \
        .drop(columns='dset') \
        .drop_duplicates()
               
    # Business features limit 
    features_business_lim = features_business[features_business['dset'].isin(dsets_list)] \
        .drop(columns='dset') 
    
    # Get NAICS nodes associated with the dset businesses
    naics_ind = edges_lim[['target']].drop_duplicates()
    
    # Limit NAICS
    features_naics_lim = features_naics.merge(naics_ind.set_index('target'), 
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