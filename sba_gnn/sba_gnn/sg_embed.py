#
# Functions relating to embeddings.
# Embeddings plot, also clustering
# Parts of this based on 
#https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors

from sklearn.manifold import TSNE
from IPython.display import display, HTML

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score

from sklearn.preprocessing import MinMaxScaler


#
# Embeddings TSNE and plot
#

def emb_tsne_transform(embed_df):
    """ TSNE transform of embeddings data """

    trans = TSNE(n_components=2)
    emb_transformed = pd.DataFrame(trans.fit_transform(embed_df), 
                                   index=embed_df.index)
    return emb_transformed

# TSNE plot, color by feature
def emb_plot(tsne_df, color_var = 'sector_num', alpha=0.7,
            figsize = (7, 7), cmap = 'jet',
            title_str = None,
            aspect = 'equal',
            outfile_folder = None,
            outfile_prefix = None,
            log_scale = False):

    fig, ax = plt.subplots(figsize=figsize)
    s = ax.scatter(
        tsne_df[0],
        tsne_df[1],
        c=tsne_df[color_var],
        cmap=cmap,
        alpha=alpha,
    )
    ax.set_xlabel("$X_1$")
    ax.set_ylabel("$X_2$")


    if title_str is not None:
        plt.title(f'{title_str}\nby {color_var}')
    else:
        plt.title(f'TSNE by {color_var}')
    
    if not log_scale:
        norm = colors.Normalize(tsne_df[color_var].min(), tsne_df[color_var].max())
    else:
        norm = colors.LogNorm(tsne_df[color_var].min() + 0.002, tsne_df[color_var].max())
    
    ax.set_aspect(aspect)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    if outfile_folder is not None:
        fig.savefig(oufile_folder.joinpath(outfile_prefix + '_' + color_var + '.png'),
                    bbox_inches='tight')
    
    return ax

#
# KMeans
#

# K Means and sillhouette information
def get_clusters_silhouettes(embed_df, n_clusters, random_state = 10):
    """ Return clusers and silhouette values """
    clusterer = KMeans(n_clusters=n_clusters, random_state=random_state,
                             n_init='auto')
    cluster_labels = clusterer.fit_predict(embed_df)
    
    silhouette_avg = silhouette_score(embed_df, cluster_labels)
    sample_silhouette_values = silhouette_samples(embed_df, cluster_labels)
    
    cluster_centers = clusterer.cluster_centers_
    
    return cluster_labels, cluster_centers, silhouette_avg, sample_silhouette_values

# Plot silhouettes
# See https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
def plot_silhouette(silhouette_values, cluster_labels, label_x_pos = -0.05,
                   cmap = cm.nipy_spectral, blank_factor = 15):
    
    n_clusters = len(np.unique(cluster_labels))
    silhouette_avg = np.mean(silhouette_values)
    
    fig, ax = plt.subplots()
    
    xmin = np.min([-0.1, np.min(silhouette_values)])
    xmax = np.max([0.2, np.max(silhouette_values)])
    if xmax > 0.6:
        xmax = 1
    label_x_pos = np.min([label_x_pos, xmax])
    ax.set_xlim([xmin, xmax])
    
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax.set_ylim([0, len(silhouette_values) + (n_clusters + 1) * blank_factor])
                   
    y_lower = 10
    for index, v in np.ndenumerate(np.unique(cluster_labels)):
        i = index[0]
        
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = silhouette_values[cluster_labels == v]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cmap(float(i) / n_clusters)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(label_x_pos, y_lower + 0.5 * size_cluster_i, str(v), fontsize = 12)

        # Compute the new y_lower for next plot
        y_lower = y_upper + blank_factor  # 10 for the 0 samples

    ax.set_xlabel("Silhouette Values", fontsize = 12)
    ax.set_ylabel("Clusters", fontsize = 12)

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    

    #ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
    return ax
    
def plot_clusters(tsne_df, labels_ser, center_label = False,
                 alpha=0.7,
                  figsize = (7, 7),
                  title_str = None,
                  aspect = 'equal',
                  cmap = cm.nipy_spectral,
                  colorbar = False):
    
    n_clusters = labels_ser.drop_duplicates().count()
    
    if pd.api.types.is_categorical_dtype(labels_ser):
        labels_ser = labels_ser.copy().cat.codes
    
    # Get min max scaled series
    mm_scaler = MinMaxScaler()
    mm_scaler.set_output(transform='pandas')
    label_float = mm_scaler.fit_transform(labels_ser.astype('float') \
                                        .to_frame()).iloc[:,0]
    colors_n = cmap(label_float)

    fig, ax = plt.subplots(figsize=figsize)
    s = ax.scatter(
        tsne_df[0],
        tsne_df[1],
        c = colors_n,
        alpha=alpha,
    )
    ax.set_xlabel("$X_1$")
    ax.set_ylabel("$X_2$")

    if center_label:
        # Labeling the clusters
        # Draw white circles at clusters
        tsne_centers = pd.concat([tsne_df, label_float.rename('cluster').to_frame()], axis=1) \
           .groupby('cluster') \
            .agg('median')
    
        ax.scatter(
            tsne_centers.iloc[:, 0],
            tsne_centers.iloc[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k")

        for index, v in np.ndenumerate(n_clusters):
            i = index[0]
            ax.scatter(tsne_centers.iloc[i, 0], tsne_centers.iloc[i, 1], 
                       marker="$%d$" % i, alpha=1, s=50, edgecolor="k")
            
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        norm = colors.Normalize(labels_ser.min(), labels_ser.max())

        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        
    ax.set_aspect(aspect)
    
    return ax 
                   
                   