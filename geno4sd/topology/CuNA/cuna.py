_author__ = "Aritra Bose"
__copyright__ = "Copyright 2022, IBM Research"
__version__ = "0.1"
__maintainer__ = "Aritra Bose"
__email__ = "a.bose@ibm.com"
__status__ = "Development"

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
sns.set(color_codes=True)

from matplotlib.pyplot import figure
from itertools import combinations
import scipy.stats as stats
from operator import itemgetter

import networkx as nx
from networkx.algorithms import community
from netgraph import Graph
import os, time


def count_list(lst, x):
    
    """
    Compute sum of an elements if it appears in the list
    """
    return sum(x in item for item in lst)
def count_two_list(lst,t):
    
    """
    Compute number of otimes two elements occur together in a list
    """
    ct = 0
    for k in range(0,len(lst)):
        if (t[0] in lst[k]) & (t[1] in lst[k]):
            ct = ct + 1
    return ct


def get_network(rdscr_df, plot_flag, pvalues, community_flag=1, verbose=0):
    
    """
    Compute the network interactions from the cumulants. 

    Parameters:
    
    -----------
    rdscr_df:
        Dataframe containing output from cumulants.py with redescription groups in rows and their respective 
        residuals, mean, standard deviation, p-value and z-scores in columns. 

    plot_flag:
        A flag when set plotting the network is performed using the networkx library. 
        Usually set to 0 as we would like to draw the network in Cytoscape. 

    pvalues:
        A list containing p-value cutoffs to be used. The code will compute stable interaction for each p-value 
        and retain all interactions which survives these cutoffs. 

    community_flag: 
        A flag when set, CuNA computes the greedy communities from the network
       
    verbose:
        verbose flag for intermediate output to stdout


    Returns:

    -----------

    graph_interactions: 
        A dataframe which can be ingested by any network producing tool such as networkx or Cytoscape.

    nodes:
        List of nodes (or vertices) of the network 

    comm_df:
        A dataframe with communities. 
        This might contain None in the fields as each community has its own size.
       
    node_rank_df: 
        A dataframe with nodes and their respective ranks in the graph G
        
    """
    if verbose == 1:
        print("\n Reading " + str(rdscr_df.shape[0]) + " cumulants \n")
        print(rdscr_df)
    
    rdscr_df.reset_index(inplace=True)
    rdscr_df.dropna(inplace=True)
    appended_df = []
    #iterate over each p-value threshold
    for p_i in pvalues:

        if verbose == 1:
            print("---- P-VALUE :"+str(p_i)+" -----")
        #filter cumulants with z-score = 3 and user-defined p-values
        filt_cumulants_df = rdscr_df.loc[(abs(rdscr_df['Z'].astype(float)) > 3) & (rdscr_df['P'].astype(float) < p_i)]
        if verbose == 1:
            print("Significant clusters: " + str(filt_cumulants_df.shape[0]))
        verts = list(filt_cumulants_df[filt_cumulants_df.columns[0]])
        #print(verts)
        #get list of all unique combinations
        res = []

        for i in range(0,len(verts)):
            a = verts[i].split('&')
            res.append(a)

        flatres = [i for sublist in res for i in sublist]
        num_nodes = len(set(flatres))
        if verbose == 1:
            print("\n Number of features: ", num_nodes)
        allcomb = list(combinations(set(flatres),2))

        sig_pairs = []
        #generate p-value for each pair
        for i in range(0, len(allcomb)):
            a = allcomb[i][0]
            b = allcomb[i][1]
            n_ab = count_two_list(res,allcomb[i])
            #print(str(a) + "---" + str(b) + " : " + str(n_ab))
            
            n_axb = count_list(res, a) - n_ab
            n_xab = count_list(res, b) - n_ab
            n_xaxb = len(res) - n_ab - n_xab - n_axb
            C = [[n_axb, n_ab], [n_xaxb,n_xab]]
            oddsratio, pval = stats.fisher_exact(C)
            if pval < 0.01:
                sig_pairs.append((a,b,pval))

        df_sigpairs = pd.DataFrame(sig_pairs, columns=["v1","v2","pval"])

        cols = ["v1","v2"]
        res = []
        for i in range(0,len(verts)):
            a = verts[i].split('&')
            if len(a) == 1:
                tmp = [verts[i],verts[i]]
                d = dict(zip(cols,tmp))
            else:
                tmp = list(combinations(a,2))
                d = [dict(zip(cols,i)) for i in tmp]
                res.append(pd.DataFrame(d))

        df = pd.concat(res,names=cols)


        #get each pair of interactions
        interactions = pd.DataFrame(df.groupby(df.columns.tolist(),as_index=False).size())
        #print(interactions)
        interactions.columns = ["v1","v2","count"]

        #merge count and pvalue dataframes
        p1 = pd.merge(df_sigpairs, interactions, on=['v1','v2'])
        interactions.columns = ["v2","v1","count"]
        p2 = pd.merge(df_sigpairs, interactions, on=['v1','v2'])

        p1p2 = p1.append(p2, ignore_index=True)
        sorted_pval = p1p2.sort_values(by="pval", ascending=True)

        if verbose == 1:
            print("\n Number of interactions: ", sorted_pval.shape[0])
        nodes = pd.unique(sorted_pval[['v1','v2']].values.ravel('K'))
        if verbose == 1:
            print("\n Number of unique nodes: ", len(nodes))

        appended_df.append(sorted_pval)

    final_df = pd.concat(appended_df)
    dupdf = final_df[final_df.duplicated(['v2', 'v1'])]

    #print("\n ====== STABLE INTERACTIONS ====== \n")
    #sort by count and extract stable interactions
    s = dupdf.groupby(by=['v1','v2']).count()

    #get robust interactions which appeared through all of the lists
    df1 = pd.DataFrame(s.index.tolist(), columns=list(['v1','v2']))

    #merge and get relevant counts
    df_ct = pd.merge(dupdf, df1, on=['v1','v2'], how='inner')

    #get stable interactions by dropping duplicates
    stable_interactions = df_ct.sort_values('count', ascending=False).drop_duplicates(['v1','v2']).sort_index()
    if verbose == 1:
        print(stable_interactions)

    #iterate over a list of thresholds and retain the minimum cutoff with all nodes
    thresnodes = num_nodes
    graph_interactions = []
    cutoff = np.linspace(0.9,0.1,12)
    for thres in cutoff:
        if thresnodes == num_nodes:

            graph_interactions = stable_interactions[:int(thres*len(stable_interactions))]
            nodes = pd.unique(graph_interactions[['v1','v2']].values.ravel('K'))
            thresnodes = len(nodes)
            fthres = thres
    
    if verbose == 1:
        print("\n Cutoff for network is: ", str(fthres))

    idx = np.where(cutoff == fthres)[0]
    graph_interactions = stable_interactions[:int(cutoff[idx-1]*len(stable_interactions))]
    nodes = pd.unique(graph_interactions[['v1','v2']].values.ravel('K'))
   
    
    G = nx.Graph()
    G = nx.from_pandas_edgelist(graph_interactions, 'v1', 'v2', edge_attr='count')
    
    if plot_flag == 1:
        draw_network(graph_interactions)
   
    comm_df = pd.DataFrame()
    if community_flag == 1:
        #obtain communities
        comm_df = get_communities(G, verbose)

    #obtain node rank
    node_rank_df = rank_nodes(G, nodes,verbose)

    return graph_interactions, nodes, comm_df, node_rank_df

def draw_network(edge_df):
    """
    Plot the networkx graph with edge size corresponding the count (weight). 

    Parameters:
    
    -----------

    G: 
        Networkx graph object

    """ 

    #Plot the Graph if flag is set
    
    fig, ax = plt.subplots(figsize=(10,8))
    pos = nx.spring_layout(G)

    count = [i['count']/50 for i in dict(G.edges).values()]
    labels = [i for i in dict(G.nodes).keys()]
    labels = {i:i for i in dict(G.nodes).keys()}

    nx.draw_networkx_nodes(G, pos, ax = ax, labels=True)
    nx.draw_networkx_edges(G, pos, width=count, ax=ax, alpha=0.75)
    _ = nx.draw_networkx_labels(G, pos, labels, ax=ax)

    fig.savefig("CuNA_Network.png", dpi=600, bbox_inches='tight')

def get_communities(G,verbose=0):

    """
    Compute the rank of each node in the graph by computing an average of different centrality 
    measures such as degree, betweenness, information, voterank, and eigenvector. 

    Parameters:

    -----------

    G: 
        Networkx graph object 

    verbose:
        Verbose flag for printing intermediate results in stdout

    Parameters:

    -----------

    comm_df:
        A dataframe with communities. 
        This might contain None in the fields as each community has its own size.
    """
    
    #perform greedy modularity communities from networkx
    communities = community.greedy_modularity_communities(G, weight='count')
    if verbose == 1:
         print("The CuNA network has " +  str(len(communities)) + " communities")
    

    modularity_dict = {}
    # Loop through the list of communities, keeping track of the number 
    for idx,comm in enumerate(communities): 
        # Loop through each node in a community
        for node in comm:
            # Create an entry in the dictionary for the node and store the value they belong to
            modularity_dict[node] = idx 

    community_dicts = {}
    partitions = []
    for i,c in enumerate(communities): # Loop through the list of communities
        #Filter out modularity classes with 2 or fewer nodes
        #if len(c) > 2: # 
        # Print out the classes and their members
        community_dicts[i] = list(c)
        partitions.append(list(c))
        if verbose == 1:
            print('Community '+str(i)+':', list(c)) 
            print('\n')

    index_vals = ['Community'+str(i) for i in range(0,len(community_dicts.keys()))]
    comm_df = pd.DataFrame.from_dict(community_dicts, orient='index')
    comm_df.index = index_vals

    return comm_df
    
    

def rank_nodes(G, nodes, verbose=0):

    """
    Compute the rank of each node in the graph by computing an average of different centrality 
    measures such as degree, betweenness, information, voterank, and eigenvector. 

    Parameters:
    
    -----------

    G: 
        Networkx graph object 

    nodes: 
        List of nodes in the graph G

    verbose:
        verbose flag to print the intermediate output to stdout

    Returns:
    
    -----------

    rankings: 
                A dataframe with nodes and their respective ranks in the graph G
    """

    #defining dictionary with nodes in keys. 
    signodes = {key: None for key in nodes}

    # degree
    degree_dict = dict(G.degree(G.nodes()))
    # betweenness centrality 
    betweenness_dict = nx.betweenness_centrality(G, weight='count') 
    # eigenvector centrality
    eigenvector_dict = nx.eigenvector_centrality(G, weight='count') 
    # voterank 
    voterank_dict = nx.voterank(G)
    # information centrality
    info_dict = nx.current_flow_closeness_centrality(G, weight='count')

    # Assign each to an attribute in your network
    nx.set_node_attributes(G, degree_dict, 'degree')
    nx.set_node_attributes(G, betweenness_dict, 'betweenness')
    nx.set_node_attributes(G, eigenvector_dict, 'eigenvector')
    nx.set_node_attributes(G, voterank_dict, 'voterank')
    nx.set_node_attributes(G, info_dict, 'information')

    #sort by the centrality measures
    sorted_degree = sorted(degree_dict.items(), key=itemgetter(1), reverse=True)
    sorted_betweenness = sorted(betweenness_dict.items(), key=itemgetter(1), reverse=True)
    sorted_eigenvec = sorted(eigenvector_dict.items(), key=itemgetter(1), reverse=True)
    sorted_info = sorted(info_dict.items(), key=itemgetter(1), reverse=True)

    #populate dictionary by appending importance of each node
    for i,d in enumerate(sorted_degree):
        signodes[d[0]] = [i+1]
    for i,tb in enumerate(sorted_betweenness):
        signodes[tb[0]].append(i+1)
    for i,tb in enumerate(sorted_eigenvec):
        signodes[tb[0]].append(i+1)
    for i,tb in enumerate(sorted_info):
        signodes[tb[0]].append(i+1)
    for i,tb in enumerate(voterank_dict):
        signodes[tb].append(i+1)

    #dictionary containing nodes as keys and rank as value
    rank_nodes = {key:None for key in nodes}
    for k,v in signodes.items():
        rank_nodes[k] = np.mean(v)

    rankings = pd.DataFrame(rank_nodes.items(), columns=['Node','Score'])
    rankings.sort_values(by=['Score'], ascending=True, inplace=False)

    if verbose == 1:
        print("Top 5 nodes in the graph")
        print(rankings[:5])

    return rankings
    
   
