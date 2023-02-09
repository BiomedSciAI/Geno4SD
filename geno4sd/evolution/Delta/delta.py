__author__ = "Kahn Rhrissorrakrai"
__copyright__ = "Copyright 2022, IBM Research"

__version__ = "0.0.1"
__maintainer__ = "Kahn Rhrissorrakrai"
__email__ = "krhriss@us.ibm.com"
__status__ = "Development"

## Citation
#The first use of Delta has been discussed in:
#Parikh, A.R., Leshchiner, I., Elagina, L. et al. Liquid versus tissue biopsy for detecting acquired resistance and tumor heterogeneity in gastrointestinal cancers. Nat Med 25, 1415–1421 (2019). https://doi.org/10.1038/s41591-019-0561-9
#Please cite the above article if you use this tool.

## Disclosures
#The following disclosure to the USPTO related to Delta:
#  Clump pattern identification in cancer patient treatment. USPTO 16/288,371, 2020.


import pandas as pd
from sklearn.cluster import SpectralCoclustering
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from geno4sd.utils import clustering_tools

class cluster_delta:
    def __init__(self, delta):
        self.delta = delta
        self.clustering = None
        self.fit_data = None


####### Utils #####

def calculate_patient_delta_predefined( df, pair_df ):
    """
    Function to calculate the Delta per predefined pair of samples

    :param df: dataframe with samples as rows and columns as features 
    :param pair_df: 2 column dataframe for the pair of sample IDs on which to calculate the difference
    :return: a dataframe where rows are pairs of samples and column is the difference between those pairs
    
    """
    df = df.fillna(0)

    # Drop patientid column if present
    if 'patientID' in df.columns:
        df = df.drop('patientID')

    deltas = []
    rnames = []
    for p in range(len(pair_df)):
        pair = pair_df.iloc[p,:]
        if pair[0] in df.index.to_list() and pair[1] in df.index.to_list():
                lead = df.loc[pair[1],:]
                lag = df.loc[pair[0],:]
                delt = lead-lag
                deltas.append( delt.to_list() )
                rnames.append( pair[0]+':'+pair[1] )
    deltas = pd.DataFrame( deltas, index = rnames, columns=df.columns)
    return deltas

def calculate_patient_delta_sliding_window_driver( df, days_dx, window_size=2):
    """
    Function to calculate Deltas over a matrix of patients

    :param df: dataframe with samples as rows and columns as features, but there is a column 'patientID'
    :param days_dx: dictionary where keys are the df indices (sample names) and the value is the date
    :param window_size: how far apart should 2 samples be. Default is 2 so samples are adjacent
    :return: dataframe where rows are pairs of samples and columns is the difference between those pairs

    """

    df = df.fillna(0)
    pat_ids = list(set( df['patientID'] ))
    pat_ids.sort()
    deltas = pd.DataFrame()
    for i in range( 0, len( pat_ids ) ):
        deltas = deltas.append( calculate_patient_delta_sliding_window( df = df,
                                                                        pat_id = pat_ids[i],
                                                                        days_dx = days_dx,
                                                                        window_size = window_size ) )

    return deltas

def calculate_patient_delta_sliding_window(df, pat_id, days_dx, window_size=2):
    """
    Function to calculate Delta and produce a matrix of sample pairs and the delta values for a given patient 
    given a window size

    :param df: dataframe with samples as rows and columns as features, but there is a column 'patientID'
    :param pat_id: id of patient of interest, contains in 'patientID' column of 'df'
    :param days_dx: dictionary where keys are the df indices (sample names) and the value is the date
    :param window_size: how far apart should 2 samples be. Default is 2 so samples are adjacent
    :return: dataframe where rows are pairs of samples and columns is the difference between those pairs

    """
    sel_mat = df[df['patientID'] == pat_id ]
    sel_mat = sel_mat.drop('patientID', axis = 1)
    days_dx_sel = []
    for x in sel_mat.index:
        if x in days_dx.keys():
            days_dx_sel.append([x, days_dx[x]])
    days_dx_sel = pd.DataFrame(days_dx_sel, columns=['sampleID', 'daysFromDx'])
    days_dx_sel = days_dx_sel.drop_duplicates()
    days_dx_sel = days_dx_sel.sort_values('daysFromDx')

    deltas = []
    deltas_r_names = []
    if len(days_dx_sel) >= window_size:
        for i in range( window_size-1, len(days_dx_sel) ):
            lead = sel_mat.loc[days_dx_sel['sampleID'].iloc[i]]
            lag = sel_mat.loc[days_dx_sel['sampleID'].iloc[i - window_size-1]]
            delt = lead - lag
            deltas.append(delt.to_list())
            deltas_r_names.append(days_dx_sel['sampleID'].iloc[i - window_size-1] + ':' + days_dx_sel['sampleID'].iloc[i])
        deltas = pd.DataFrame(deltas, index=deltas_r_names, columns=sel_mat.columns)
        return deltas
    else:
        return None


# Plot heatmap of cluster delta
def plot_cluster_patient_delta(cluster_res, title = '', savefile = ''):
    """
    Function to plot heatmap of the clustered deltas

    :param cluster_res: clustered delta object
    :param title: string for plot title
    :param savefile: path for filename to save plot (optional)

    """
    delta_df = cluster_res.delta
    rowClusterMap = dict(zip( list(delta_df.index), cluster_res.clustering.row_labels_))
    colClusterMap = dict(zip( list(delta_df.columns), cluster_res.clustering.column_labels_))

    label_rows = [ 'r' + str(rowClusterMap[x]) for x in cluster_res.fit_data.index ]
    label_cols = [ 'c' + str(colClusterMap[x]) for x in cluster_res.fit_data.columns ]

    row_cols = sns.color_palette("hls", len(set(label_rows ) ) )
    row_lut = dict(zip( list(set(label_rows )), row_cols))
    row_colors = pd.Series(label_rows).map(row_lut)

    col_cols = sns.color_palette("hls", len(set(label_cols ) ) )
    col_lut = dict(zip( list(set(label_cols )), col_cols))
    col_colors = pd.Series(label_cols).map(col_lut)

    plt.figure(figsize = (10,10))
    g = sns.clustermap(cluster_res.fit_data.reset_index(drop=True).transpose().reset_index(drop=True).transpose(),
                       row_colors=row_colors, 
                       col_colors=col_colors,
                       row_cluster=False, 
                       col_cluster=False, 
                       cmap='vlag', 
                       center = 0)
    plt.title( title )

    label_rows = list(set(label_rows))
    label_rows.sort()
    label_cols = list(set(label_cols))
    label_cols.sort()
    for label in label_rows:
        g.ax_row_dendrogram.bar(0, 0, color=row_lut[label],
                                label=label, linewidth=0)
    g.ax_row_dendrogram.legend(loc="center", ncol=1)

    for label in label_cols:
        g.ax_col_dendrogram.bar(0, 0, color=col_lut[label],
                                label=label, linewidth=0)
    g.ax_col_dendrogram.legend(loc="center", ncol=len(label_cols))

    g.cax.set_position([-.05, .2, .03, .45])

    if savefile != '':
        plt.savefig(savefile)
    plt.show()
    plt.close()

def cluster_patient_delta( cluster_res, cluster_model = None, n_cluster = 10 ):
    """
    Function to cluster the delta values

    :param cluster_res: cluster_delta object
    :param cluster_model: scikit clustering model to be used. Default is SpectralCoClustering
    :param n_cluster: integer specifying number of clusters to look for
    :return: cluster_delta object with clustered results
    """
    delta_df = cluster_res.delta

    # If python clustering object not specified, then set to default biclustering method
    if cluster_model is None:
        cluster_model = SpectralCoclustering(n_clusters=n_cluster, random_state=0)

    res_cluster = cluster_model.fit(delta_df)
    fit_data = delta_df.loc[list(delta_df.index[np.argsort(res_cluster.row_labels_)]),
                            list(delta_df.columns[np.argsort(res_cluster.column_labels_)])]

    cluster_res.clustering = res_cluster
    cluster_res.fit_data = fit_data

    return(cluster_res)


def calculate_patient_delta( df, pair_df = None, days_dx = None, window_size = 2, mode = 'predefined',
                             cluster_model = None, n_cluster = None, plot = False, savefile = '' ):
    """
    Function to perform the full delta calculation and clustering

    :param df: dataframe with samples as rows and columns as features, but there is a column 'patientID'
    :param pair_df: 2 column dataframe for the pair of sample IDs on which to calculate the difference. Required if mode = 'predefined'
    :param days_dx: dictionary where keys are the df indices (sample names) and the value is the date
    :param window_size: how far apart should 2 samples be. Default is 2 so samples are adjacent
    :param mode: delta mode to calculte ['predefined', 'sliding']. Default is 'predefined' for pair of samples. 'sliding' enables specifying distance between patient samples to consider.
    :param cluster_model: scikit clustering model to be used. Default is SpectralCoClustering
    :param n_cluster: integer specifying number of clusters to look for [None, int, list(int)]. If None, then eigengap approach used to identify optimal number of clusters. Otherwise can specify integer of number of clusters or list of integers for number of clusters of interest.
    :param plot: boolean whether to plot heatmap.
    :param savefile: path for filename to save plot (optional)
    :return: cluster_delta object

    """
    # Run Delta in specified mode
    if (mode == 'predefined') & (pair_df is not None):
        res = calculate_patient_delta_predefined( df = df, pair_df = pair_df )
    elif (mode =='sliding'):
        res = calculate_patient_delta_sliding_window_driver( df = df, days_dx = days_dx, window_size=window_size)
        df = df.drop('patientID',axis=1)

    # Save results
    results = cluster_delta(res)

    # Cluster the delta
    if n_cluster == None:
        affinity_matrix = clustering_tools.getAffinityMatrix(df, n_cluster = 20)
        k, _,  _ = clustering_tools.eigenDecomposition(affinity_matrix, plot = plot)
        k.sort()
        n_cluster = k[ k >=3 ][0]
        results = cluster_patient_delta(cluster_res = results, 
                                            cluster_model = cluster_model, 
                                            n_cluster = n_cluster )
        if plot:
            plot_cluster_patient_delta(results,
                                       title = 'K =' + str(n_cluster),
                                       savefile = savefile)
    elif isinstance(n_cluster, int):
        results = cluster_patient_delta(cluster_res = results, 
                                            cluster_model = cluster_model, 
                                            n_cluster = n_cluster )
        if plot:
            plot_cluster_patient_delta(results,
                                       title = 'K =' + str(n_cluster),
                                       savefile = savefile)
    elif isinstance(n_cluster, list):
        cluster_res_by_n = dict()
        for n in n_cluster:
            cluster_res_by_n[n] = cluster_patient_delta(cluster_res = results, 
                                                                      cluster_model = cluster_model, 
                                                                      n_cluster = n )
            if plot:
                plot_cluster_patient_delta(cluster_res_by_n[n],
                                        title = 'K =' + str(n),
                                        savefile = savefile)
        results = cluster_res_by_n

    return (results)
