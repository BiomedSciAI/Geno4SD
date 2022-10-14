__author__ = "Filippo Utro"
__copyright__ = "Copyright 2022, IBM Research"

__version__ = "0.0.1"
__maintainer__ = "Filippo Utro"
__email__ = "futro@us.ibm.com"
__status__ = "Development"

import logging
import math
import os
from random import shuffle
import random
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def _compute_count_in_region(sample_data, start, stop):
    a = sample_data[(sample_data['start'] >= start) & (sample_data['stop'] <= stop)]  # fully contained
    b = sample_data[(sample_data['stop'] > start) & (sample_data['stop'] <= stop)]  # partially contained
    c = sample_data[(sample_data['start'] >= start) & (sample_data['start'] <= stop)]  # partially contained
#   d = sample_data[(sample_data['start'] >= start) & (sample_data['stop'] >= stop) & (sample_data['start'] <= stop)]  # partially contained
    a = pd.concat([a, b, c]).drop_duplicates() #.append(d)
    return a.groupby('samples').count().reset_index()


def compute_mutational_load(sample_data, samples, regions_size, chromosome, window_size=-1, out_file_name=None):
    """
        Function to compute the mutational load files

        Parameters:
        -----------
        sample_data: 
            panda dataframe containing the sample with relative start and stop alteration
        samples: 
            list of all samples
        regions_size: 
            panda dataframe containing the start and stop of the region of interest
        chromosome: int
            relative to the chromosome of interest
        window_size: int
            relative the the window size, by default the entire region is used. Note if the window size is not multiple of the region size the last window will be the remaining region portion.
        out_file_name: str, optional
            if provided the mutational load table is stored as csv file.
        
        Returns:
        --------
        a dataframe containing the mutational load, each row is a sample, the column is a window
    """
    mutation_load_data = pd.DataFrame()
    for index_region, row_region in regions_size.iterrows():
        start = row_region['start']
        stop = row_region['stop']

        if window_size > 0:
            end = start + window_size
            for bin_window in range(0, math.ceil((stop-start)/window_size)):
                end = end + (bin_window * window_size)
                if end>stop:
                    end = stop
                count_in_window = _compute_count_in_region(sample_data, start, end)
                mutation_load_data_bin = pd.DataFrame()
                mutation_load_data_bin['samples'] = count_in_window['samples']
                mutation_load_data_bin['count'] = count_in_window['start']
                missing_sample_in_window = set(samples).difference((set(mutation_load_data_bin['samples'])))
                tmp_data = pd.DataFrame({'samples': list(missing_sample_in_window),
                                         'count': [0]*len(missing_sample_in_window)})
                
                mutation_load_data_bin = pd.concat([mutation_load_data_bin,tmp_data])
                mutation_load_data_bin['window'] = [(chromosome, start, stop, bin_window)] * len(mutation_load_data_bin)
                mutation_load_data = pd.concat([mutation_load_data,mutation_load_data_bin])
                start = end
        else:
            count_in_window = _compute_count_in_region(sample_data, start, stop)
            mutation_load_data_region = pd.DataFrame()
            mutation_load_data_region['samples'] = count_in_window['samples']
            mutation_load_data_region['count'] = count_in_window['start']
            missing_sample_in_window = set(samples).difference((set(mutation_load_data_region['samples'])))
            tmp_data = pd.DataFrame({'samples': list(missing_sample_in_window),
                                     'count': [0] * len(missing_sample_in_window)})
            mutation_load_data_region = pd.concat([mutation_load_data_region, tmp_data])
            mutation_load_data_region['window'] = [(chromosome, start, stop, -1)] * len(mutation_load_data_region)
            mutation_load_data = pd.concat([mutation_load_data, mutation_load_data_region])
    
    mutation_load_pivot = mutation_load_data.sort_values(['samples', 'window']).drop_duplicates().pivot('samples', 'window', 'count').fillna(0)
   
    if out_file_name is not None:
        mutation_load_pivot.to_csv(out_file_name)
    return mutation_load_pivot


def _compute_shingle(mutation_load, list_samples, id_sample, moment_type=1):
    unique, counts = np.unique(list_samples, return_counts=True)
    tmp = mutation_load.loc[unique, :].copy()
    for i in range(0, len(counts)):
        if counts[i] > 1:
            tmp = tmp.append(mutation_load.loc[unique[i], :], sort=False)
    
    if (moment_type == 1):
        moment1 = pd.DataFrame(tmp.mean())
    elif (moment_type == 2):
        moment1 = pd.DataFrame(tmp.var())
    elif (moment_type == 3):
        moment1 = pd.DataFrame(tmp.skew())
    else:
        moment1 = pd.DataFrame(tmp.kurt())
    moment1.columns = [id_sample]

    return moment1


def compute_shingle(train_samples, test_samples,mutation_load, moment_type=1,  out_file_name=None):
    """
        This function compute the shingles, generating the train and test sample
        
        Parameters:
        -----------
        rain_samples: 
            a panda dataframe containing the train ids to use to create the shingle
        test_samples: 
            a panda dataframe containing the test ids to use to create the shingle
        mutation_load: 
            mutation load from which compute the shingle
        moment_type: int, optional, default=1, value=[1,4]
            the moments of a function are quantitative measures related to the shape of the distribtion. 
        out_file_name: str 
            indicating the filename to store the shingle in csv if provided, default=None
        
        Returns:
        --------
        a panda datagrame containing the shingles

        References
        ----------
        Parida L, Haferlach C, Rhrissorrakrai K, Utro F, Levovitz C, Kern W, et al. (2019) 
        Dark-matter matters: Discriminating subtle blood cancers using the darkest DNA. 
        PLoS Comput Biol 15(8): e1007332. https://doi.org/10.1371/journal.pcbi.1007332
    """
    
    shingles = pd.DataFrame()

    for key, samples_in_train in train_samples.items():
        id_sample = 'Train_'+str(key[1])+'_fold_'+str(key[0])+'_'+key[2]
        moment1 = _compute_shingle(mutation_load, samples_in_train, id_sample, moment_type)
        shingles = shingles.append(moment1.T)

    for key, samples_in_test in test_samples.items():
        id_sample = 'Test_'+str(key[1])+'_fold_'+str(key[0])+'_'+key[2]
        moment1 = _compute_shingle(mutation_load, samples_in_test, id_sample, moment_type)
        shingles = shingles.append(moment1.T)

    if out_file_name is not None:
        shingles.to_csv(out_file_name)

    return shingles

def permute_labels(label_info, seed=None):
    """
        Permute the sample labels.

        Parameters:
        -----------
        label_info: 
            dataframe containing the original sample labels per class
        seed: int, optional, default=None
            It affects the ordering of the labels, which controls the randomness. 
            Pass an int for reproducible output across multiple function calls.
    
        Returns:
        --------
        dataframe with the permuted sample label
    """

    if seed is not None:  # None is the default
        random.seed(seed)
    
    tmp_phenotype = label_info['phenotype'].tolist()
    shuffle(tmp_phenotype)
    label_info['phenotype'] = tmp_phenotype

    return label_info

def train_test_split(label_info, test_train_sizes, num_fold, sample_size, permuted=False, seed=None):
    """
        Split data into training and test set.    

        Parameters:
        -----------
        label_info: 
            panda dataframe containing the original sample labels per class
        test_train_sizes: 
            dataframe containing the size of train and test samples for each label
        num_fold: int 
            number of folds for the crossvalidation
        sample_size: int
            number of sample used to generate a shingle.
        permuted: Boolean, optional, default=False
            If True labels will be permuted.
        seed: int, optional, default=None
            It affects the ordering of the labels, which controls the randomness. 
            Pass an int for reproducible output across multiple function calls.
        
        Returns
        -------
        train_samples : a panda dataframe containing the training ids
        test_samples :  a panda dataframe containing the testing ids
    """
    train_samples = {}
    test_samples = {}

    if seed is not None:  # None is the default
        np.random.seed(seed)

    for fold in range(0,num_fold):
        if permuted:
            logging.info('Permuting labels')
            label_info = permute_labels(label_info, seed=seed)
        for index_tr, row_tr in test_train_sizes.iterrows():
            size_train = row_tr['Train']
            size_test = row_tr['Test']
            samples = label_info[label_info['phenotype'] == row_tr['phenotype']]['samples'].tolist()
            for i in range(0, size_train):
                selected_samples = np.random.choice(samples, sample_size, replace=True)
                train_samples[(fold, i, row_tr['phenotype'])] = selected_samples
            for i in range(0, size_test):
                selected_samples = np.random.choice(samples,sample_size, replace=True)
                test_samples[(fold, i, row_tr['phenotype'])] = selected_samples
    
    return train_samples, test_samples

def compute(sample_info, label_info, test_train_sizes, regions, chromosomes, num_fold, sample_size, out_folder, window_size=-1,  n_jobs=22, moment_type=1, permuted=False, seed=None):
    """
        This function compute the mutational load and shingles, storing them in a folder as csv file
        
        Parameters:
        -----------
        sample_info: 
            panda dataframe containing the sample with relative start and stop alteration
        label_info: 
            panda dataframe containing the original sample labels per class
        test_train_sizes: 
            panda dataframe containing per each label the number of train and test sample per fold to be generated
        regions: 
            panda dataframe containing the start and stop of the region of interest
        chromosomes: 
            list containing the desidered chromosomes to compute the shingles
        num_fold: int
            number of folds for the crossvalidation
        sample_size: int
            number of element to sample to create the shingle
        out_folder: str
            name of the folder where the output will be stored, if it doesn't exist will be automatically created.
        window_size: int
            relative the the window size, by default the entire region is used. Note if the window size is not multiple of the region size the last window will be the remaining region portion.
        n_jobs: int, optional, default=22 
            number of jobs for multiprocessing
        moment_type: int, optional, default=1, value=[1,4]
            the moments of a function are quantitative measures related to the shape of the distribtion.  
        permuted: Boolean, optional, default=False
                If True labels will be permuted.
        seed: int, optional, default=None
            It affects the ordering of the labels, which controls the randomness. 
            Pass an int for reproducible output across multiple function calls.

        Returns
        -------
        a numpy array with the shingle of all chromosomes (note they chromosome order may not be guaranteed)

        References
        ----------
        Parida L, Haferlach C, Rhrissorrakrai K, Utro F, Levovitz C, Kern W, et al. (2019) 
        Dark-matter matters: Discriminating subtle blood cancers using the darkest DNA. 
        PLoS Comput Biol 15(8): e1007332. https://doi.org/10.1371/journal.pcbi.1007332
    """

    if not os.path.exists(out_folder):
        os.makedirs(out_folder, exist_ok=True)

    Parallel(n_jobs=n_jobs, verbose=5)(delayed(compute_mutational_load)(sample_info[sample_info['chr'] == chr_interest], sample_info['samples'].unique(),
                                                                  regions[regions['chr'] == chr_interest],
                                                                   chr_interest, window_size=window_size, out_file_name=os.path.join(out_folder, 'mutation_load'+str(chr_interest)+'.csv')
                                                                  ) for chr_interest in sample_info['chr'].unique())

    train_samples, test_samples = train_test_split(label_info, test_train_sizes, num_fold, sample_size, permuted=permuted, seed=seed)

    r = Parallel(n_jobs=n_jobs, verbose=5)(delayed(compute_shingle)(train_samples, test_samples, pd.read_csv(os.path.join(out_folder, 'mutation_load'+str(chr_interest)+'.csv'), index_col=0),
                                                                    moment_type,  out_file_name=os.path.join(out_folder, 'shingle_'+str(chr_interest)+'.csv')
                                                                   ) for chr_interest in chromosomes)
    
    return r