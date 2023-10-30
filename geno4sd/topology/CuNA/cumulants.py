_author__ = "Aritra Bose"
__copyright__ = "Copyright 2022, IBM Research"
__version__ = "0.1"
__maintainer__ = "Aritra Bose"
__email__ = "a.bose@ibm.com"
__status__ = "Development"

from distutils.command.sdist import sdist
import pandas as pd 
import numpy as np 
import scipy as sp 
import scipy.stats as ss
import sys, math 
import multiprocessing as mp
import time, os, subprocess
from itertools import combinations
featureNumSwitchToParallel = 120

def tupleKeyToString( tup ):
    """
    Convert tuple of indices to original index labels
    """
    return '&'.join( [ indx_labels[t] for t in list( tup ) ] )
 
def permute_columns(x):
    """
    Permute columns of a 2-d numpy array
    """
    ix_i = np.random.sample(x.shape).argsort(axis=0)
    ix_j = np.tile(np.arange(x.shape[1]), (x.shape[0], 1))
    return x[ix_i, ix_j]

def AllcumulantSingle(args):
    """
    Function to calculate cumulant for a specific combination and do the permutation testing. 
    Returns a list of lists with the mean, standard deviation and Z-scores for each test.
    """
    indx, max_combo, n_iter = args

    if max_combo:
        # computing higher order cumulant and accounting for lower order groups (removing / subtracting)
        actual = [tupleKeyToString(indx), 
                  np.prod( dat[:,indx], axis = 1 ) - ( res[tupleKeyToString((indx[0], indx[1]))] * res[tupleKeyToString((indx[2], indx[3]))] ) - ( res[tupleKeyToString((indx[0], indx[2]))] * res[tupleKeyToString((indx[1], indx[3]))] ) - ( res[tupleKeyToString((indx[0], indx[3]))] * res[tupleKeyToString((indx[1], indx[2]))] ), 
                     np.mean( np.prod( dat[:,indx], axis = 1 ) ) - ( res[tupleKeyToString((indx[0], indx[1]))] * res[tupleKeyToString((indx[2], indx[3]))] ) - ( res[tupleKeyToString((indx[0], indx[2]))] * res[tupleKeyToString((indx[1], indx[3]))] ) - ( res[tupleKeyToString((indx[0], indx[3]))] * res[tupleKeyToString((indx[1], indx[2]))] )]
        
       
    else:
        # computing lower order cumulants
        actual = [tupleKeyToString(indx), np.prod( dat[:,indx], axis = 1 ), np.mean( np.prod( dat[:,indx], axis = 1 ) ) ]

        
    rndRes = []
    for n in range(n_iter):
        datTmp = permute_columns(dat[:,indx])
        if max_combo:
            rndRes.append( np.mean( np.prod( datTmp, axis = 1 ) ) - ( np.mean( np.prod( datTmp[:,[0,1]], axis = 1 ) ) * np.mean( np.prod( datTmp[:,[2,3]], axis = 1 ) ) ) - ( np.mean( np.prod( datTmp[:,[0,2]], axis = 1 ) ) * np.mean( np.prod( datTmp[:,[1,3]], axis = 1 ) ) ) - ( np.mean( np.prod( datTmp[:,[0,3]], axis = 1 ) ) * np.mean( np.prod( datTmp[:,[1,2]], axis = 1 ) ) ) )
        else:
            rndRes.append( np.mean( np.prod( datTmp, axis = 1 ) ) )

    mn = np.mean(rndRes)
    sd = np.std(rndRes)
    if sd <  1e-12:
        sd = 0
        z = 0
    else:
        z = (actual[2]-mn)/sd
    actual += [ mn, sd, z ]
    # actual += [ 0, 0, 0 ]
    
    return [actual]

def convert_to_dict(res):
    res = [x for sublist in res for x in sublist]
    res_dict_vec = {} 
    res_dict_mean = {}
    for x in res: 
        res_dict_vec[x[0]] = x[1]
        res_dict_mean[x[0]] = x[2:]
    res_mean_df = pd.DataFrame.from_dict(res_dict_mean, orient='index').reset_index()
    #print(res_mean_df)
    res_mean_df.columns = ['k','v', 'mn', 'sd', 'z']
    
    res_vec_df = pd.DataFrame.from_dict(res_dict_vec, orient='index').reset_index()
    res_vec_df.columns = ['k'] + list(res_vec_df.columns[1:])
    
    return res_mean_df, res_vec_df

def julia_vectors( args ):

    """
    Compute tensors from Julia implementation. 
    """

    elem, dat, dat_df, s_order, order = args
    column_names = elem.split('&')
    column_indices = [dat_df.columns.get_loc(col) for col in column_names]
    vec = np.prod( dat[:, column_indices], axis = 1 ).reshape((dat.shape[0],1))
    if order == 4:
        vec = (vec 
            - s_order.loc[s_order['k'] == column_names[0]+'&'+column_names[1], 'v'].values[0] * s_order.loc[s_order['k'] == column_names[2]+'&'+column_names[3], 'v'].values[0]
            - s_order.loc[s_order['k'] == column_names[0]+'&'+column_names[2], 'v'].values[0] * s_order.loc[s_order['k'] == column_names[1]+'&'+column_names[3], 'v'].values[0]
            - s_order.loc[s_order['k'] == column_names[0]+'&'+column_names[3], 'v'].values[0] * s_order.loc[s_order['k'] == column_names[1]+'&'+column_names[2], 'v'].values[0])

    return vec

def all_julia_vectors( args ):

    """
    Compute tensors from Julia implementation. 
    """

    dat_df, dat, n_threads, second_order_df, main_order, all_labels = args

    first_order_labels = all_labels[0]

    second_order_labels = all_labels[1]

    third_order_labels = all_labels[2]

    if main_order == 4:
        fourth_order_labels = all_labels[3]

    first_order_vecs = np.array([])
    for elem in first_order_labels:
        column_names = elem.split('&')
        column_indices = [dat_df.columns.get_loc(col) for col in column_names]
        vec = np.prod( dat[:, column_indices], axis = 1 ).reshape((dat.shape[0],1))
        # vec = np.asarray(Main.julia_prod(dat[:, column_indices])).reshape((dat.shape[0],1))
        # make sure to transpose output to match AB code ( first_order_vecs == res_vec_df.T )
        first_order_vecs = np.concatenate((first_order_vecs, vec), axis = 1) if first_order_vecs.size else vec
    first_order_vecs = pd.DataFrame(first_order_vecs, columns=first_order_labels)

    second_order_vecs = np.array([])
    for elem in second_order_labels:
        column_names = elem.split('&')
        column_indices = [dat_df.columns.get_loc(col) for col in column_names]
        vec = np.prod( dat[:, column_indices], axis = 1 ).reshape((dat.shape[0],1))
        # vec = np.asarray(Main.julia_prod(dat[:, column_indices])).reshape((dat.shape[0],1))
        second_order_vecs = np.concatenate((second_order_vecs, vec), axis = 1) if second_order_vecs.size else vec
    second_order_vecs = pd.DataFrame(second_order_vecs, columns=second_order_labels)

    pool = mp.Pool(n_threads)
    order = 3
    results = pool.map( julia_vectors, [ (elem, dat, dat_df, second_order_df, order) for elem in third_order_labels ] )
    pool.close()
    third_order_vecs = pd.DataFrame(np.squeeze(results, axis=2).T, columns=third_order_labels)

    if main_order == 4:
        pool = mp.Pool(n_threads)
        order = 4
        results = pool.map( julia_vectors, [ (elem, dat, dat_df, second_order_df, order) for elem in fourth_order_labels ] )
        pool.close()
        fourth_order_vecs = pd.DataFrame(np.squeeze(results, axis=2).T, columns=fourth_order_labels)

        julia_cumulants_vecs = pd.concat([first_order_vecs, second_order_vecs, third_order_vecs, fourth_order_vecs], axis=1)

    else:

        julia_cumulants_vecs = pd.concat([first_order_vecs, second_order_vecs, third_order_vecs], axis=1)

    julia_vec_df = julia_cumulants_vecs.T
    julia_vec_df.reset_index(inplace=True)
    julia_vec_df = julia_vec_df.rename(columns={"index": "k"})
    julia_vec_df = julia_vec_df.fillna(0)

    return julia_vec_df

def julia_cumulants( args ):

    """
    Compute cumulants using Julia implementation . 
    """

    global julia_res
    julia_res = []
    julia_df = pd.DataFrame()
    dat, dat_df, n_threads, order, verbose = args

    # needed for julia vector downstream
    n, m = dat_df.shape
    f = math.factorial
    k = f(m) // f(2) // f(m-2)

    separator = "/"
    parts = __file__.split(separator)
    cuna_dir = separator.join(parts[:-1]) + separator

    np.save(cuna_dir + 'julia_dat.npy', dat)

    command = "export JULIA_NUM_THREADS=8"
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    command = "julia " + cuna_dir + "cumulants.jl " + cuna_dir + " " + str(order)
    if verbose == 1:
        print("running:", command)
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if verbose == 1:
        if result.returncode == 0:
            print("Command executed successfully.")
            # print("Output:")
            print(result.stdout)
        else:
            print("Command failed with an error.")
            print("Error Output:")
            print(result.stderr)

    first_order_labels = indx_labels.values()
    second_order_labels = ["&".join(a) for a in combinations(indx_labels.values(), 2)]
    third_order_labels = ["&".join(a) for a in combinations(indx_labels.values(), 3)]
    if order == 4:
        fourth_order_labels = ["&".join(a) for a in combinations(indx_labels.values(), 4)]
        julia_df['k'] = list(first_order_labels) + second_order_labels + third_order_labels + fourth_order_labels
        all_labels = [first_order_labels, second_order_labels, third_order_labels, fourth_order_labels]
    else:
        julia_df['k'] = list(first_order_labels) + second_order_labels + third_order_labels
        all_labels = [first_order_labels, second_order_labels, third_order_labels]

    julia_out = np.load(cuna_dir+"julia_cumulants.npz")
    x = list(julia_out.files)
    x.sort()
    for elem in x:
        julia_df[elem] = julia_out[elem]

    julia_df = julia_df.rename(columns={'arr_0': 'v', 'arr_1': 'mn', 'arr_2': 'sd', 'arr_3': 'z'})

    julia_vec_df = all_julia_vectors( (dat_df, dat, n_threads, julia_df.iloc[m:k+m, :], order, all_labels) )

    return julia_df, julia_vec_df
 
def AllcumulantCalcParallel( dat, dat_df, n_combo, indices, n_threads, n_iter = 50, julia = 0, order = 4, verbose = 0, newRun = True, non_max_combo = None, max_combo = None ):

    """
    Compute cumulants with multithreading. 
    """

    # Get up to the max num combo of mean products
    global res
    res = [] 
    
    if newRun:
        non_max_combo = []
        max_combo = []

    if julia:
        # testing Julia cumulants
        if verbose == 1:
            print("computing cumulants (Julia)...")
        julia_df, julia_vec_df = julia_cumulants( (dat, dat_df, n_threads, order, verbose) )

        return (julia_df, julia_vec_df)

    else:
        if verbose == 1:
            print("computing cumulants (Python)...")
        pool = mp.Pool(n_threads)
        for n in range(1,n_combo):
            if newRun:
                comb = combinations( indices, n )
                non_max_combo.append( comb )
            else:
                comb = non_max_combo[n-1]
            results = pool.map( AllcumulantSingle, [ (indx, False, n_iter) for indx in comb ] )
            res += [ r for r in results ]
        pool.close()

        res_mean_df, res_vec_df = convert_to_dict(res)    
        res = dict( zip( list( res_mean_df['k'] ), list( res_mean_df['v'] )))
                            
        # calculate the max combo cumulant
        pool = mp.Pool(n_threads)
        if newRun:
            comb = combinations( indices, n_combo )
            max_combo = comb
        else:
            comb = max_combo
        results = pool.map( AllcumulantSingle, [ (indx, True, n_iter) for indx in comb ] )
        pool.close()
        
        resTmp = []
        resTmp += [ r for r in results]
        
        res_mean_tmp, res_vec_tmp = convert_to_dict(resTmp)
        res_mean_df = pd.concat( [ res_mean_df, res_mean_tmp ])
        res_vec_df = pd.concat( [ res_vec_df, res_vec_tmp ] )
                            
        return (res_mean_df, res_vec_df) #, nonmax_combo, max_combo )

# Main Function to calculate cumulants
def get_cumulants(feature_matrix, 
                 n_iter = 50, 
                 n_threads = 25, 
                 combo_size = 4,
                 verbose = 0,
                 julia = 0,
                 order = 4):

    """
    Compute cumulants of the feature space of d-dimensional features contained in the mean-centered feature matrix.
    It compute cumulants and generates a data frame with redescription groups and their corresponding residual, mean, 
    standard deviation, p-value and z scores.
    
    Parameters:
    
    -----------

    feature_matrix: 
                    Dataframe with individuals in rows and features in columns. 
                    The cumulants are computed across the feature space.
    n_iter:
                    Number of iterations indicating the number of times permutation 
                    tests are done for statistical significance of each redescription group. 
    n_threads:
                    Number of threads used in computing cumulants. 
                    This parameter needs to be balanced to ensure effective computation vs. huge allocation of resources. 
    combo_size:
                    Number of maximum combinations used in computing cumulants in redescription groups. 
                    This parameter governs the computational complexity.
                    If there are n features then the number of cumulants computed is n^combo_size. 
                    This leads to an exponential growth.
    verbose:
                    Verbose flag for printing intermediate output to stdout
    julia:
                    Flag to compute cumulants using Julia or Python (1 or 0)

    Returns:
    
    -----------

    Dataframe with redescription groups and their corresponding residual values, mean, standard deviation, 
    p-value and z scores.

    """
    global indx_labels
    global dat
    global indices
    global n_combo
    global k_res
    global non_max_combo
    global max_combo
    n_combo = combo_size
    
    #mean-center feature matrix
    mu = feature_matrix.mean(axis=0)
    sd = feature_matrix.std(axis=0)
    #check for columns with non-zero standard deviation
    colsnan = [x for x in sd.index if abs(sd[x] > 1e-8)]
    X = feature_matrix[colsnan]
    mu = mu[colsnan]
    sd = sd[colsnan]
    X = (X-mu)/sd 
    
    # Treating the dataframe as a numpy array speeds things up a bit. So storing the feature names to the index in a dictionary to map back later
    indx_labels = dict(zip(list(range(0,X.shape[1])), list(X.columns)))
    indices = list(indx_labels.keys())
    indices.sort()
    dat = X.to_numpy()

    # Calculate Cumulants
    # if verbose == 1:
    #     print("\nComputing cumulants\n")
    #start = time.time()
    k_res = {}
    mean_df, vec_df = AllcumulantCalcParallel( dat, X, n_combo, indices, n_threads, n_iter, julia, order, verbose )
    # if verbose == 1:
    #     print( len(mean_df))
    #end = time.time()
    #print((end - start)/60)

    # if verbose == 1:
    #     print("\nComputing Cumulants final z\n")
    mean_df.columns = [ 'index', 'k_res', 'Mean','StdDev','Z' ]
    mean_df.set_index( 'index', inplace = True)
#     if verbose == 1:
#         print(ResDF)
    ResDF = mean_df.fillna(0)
    ResDF['P'] = 2.0 * ss.norm.sf(np.abs(ResDF['Z']))
    
    return ResDF, vec_df 
