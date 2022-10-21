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
import time
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


def cumulantSingle(args):
    """
    Function to calculate cumulant for a specific combination and do the permutation testing. 
    Returns a list of lists with the mean, standard deviation and Z-scores for each test.
    """
    indx, max_combo, n_iter = args
    if max_combo:
        actual = [tupleKeyToString(indx), np.mean( np.prod( dat[:,indx], axis = 1 ) ) - ( res[tupleKeyToString((indx[0], indx[1]))] * res[tupleKeyToString((indx[2], indx[3]))] ) - ( res[tupleKeyToString((indx[0], indx[2]))] * res[tupleKeyToString((indx[1], indx[3]))] ) - ( res[tupleKeyToString((indx[0], indx[3]))] * res[tupleKeyToString((indx[1], indx[2]))] ) ]
    else:
        actual = [tupleKeyToString(indx), np.mean( np.prod( dat[:,indx], axis = 1 ) )]

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
        z = (actual[1]-mn)/sd
    actual += [ mn, sd, z ]
    return actual


def cumulantCalcParallel( dat, n_combo, indices, n_threads, n_iter = 50, newRun = True, non_max_combo = None, max_combo = None ):

    """
    Compute cumulants with multithreading. 
    """

    # Get up to the max num combo of mean products
    global res
    res = [] 

    if newRun:
        non_max_combo = []
        max_combo = []

    pool = mp.Pool(n_threads)
    for n in range(1,n_combo):
        if newRun:
            comb = combinations( indices, n )
            non_max_combo.append( comb )
        else:
            comb = nonmax_combo[n-1]
        results = pool.map( cumulantSingle, [ (indx, False, n_iter) for indx in comb ] )
        res += [ r for r in results ]
    pool.close()
    resDF = pd.DataFrame(res, columns = ['k','v', 'mn', 'sd', 'z'])
    res = dict( zip( list( resDF['k'] ), list( resDF['v'] )))


    # calculate the max combo cumulant
    pool = mp.Pool(n_threads)
    if newRun:
        comb = combinations( indices, n_combo )
        max_combo = comb
    else:
        comb = max_combo
    results = pool.map( cumulantSingle, [ (indx, True, n_iter) for indx in comb ] )
    pool.close()
    resTmp = []
    resTmp += [ r for r in results ]
    resTmp = pd.DataFrame(resTmp, columns = ['k','v', 'mn', 'sd', 'z'])
    resDF = pd.concat( [ resDF, resTmp ])

    return ( resDF )#, nonmax_combo, max_combo )

# Main Function to calculate cumulants
def getCumulants(feature_matrix, 
                 n_iter = 50, 
                 n_threads = 50, 
                 combo_size = 4,
                 verbose = 0):

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
    if verbose == 1:
        print("\nComputing cumulants\n")
    #start = time.time()
    k_res = {}
    ResDF = cumulantCalcParallel( dat, n_combo, indices, n_threads, n_iter )
    if verbose == 1:
        print( len(ResDF))
    #end = time.time()
    #print((end - start)/60)

    if verbose == 1:
        print("\nComputing Cumulants final z\n")
    ResDF.columns = [ 'index', 'k_res', 'Mean','StdDev','Z' ]
    ResDF.set_index( 'index', inplace = True)
    if verbose == 1:
        print(ResDF)
    ResDF = ResDF.fillna(0)
    ResDF['P'] = 2.0 * ss.norm.sf(np.abs(ResDF['Z']))

    return ResDF 
