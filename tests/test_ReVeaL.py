"""
Test cases for MAP funtions when raw absolute data are provided
Test cases can be run with the following:
  nosetests -v --with-spec --spec-color
  coverage report -m
"""

import os
import pandas as pd
import logging
import geno4sd.ml_tools.ReVeaL as ReVeaL 
from unittest import TestCase
from pandas._testing import assert_frame_equal

logging.disable(logging.CRITICAL)

def test_permute_labels():
    label_info = pd.DataFrame()
    label_info['samples'] =['s1', 's2', 's3', 's4']
    label_info['phenotype'] =['label1', 'label1', 'label2', 'label3']

    returned_df = ReVeaL.permute_labels(label_info, seed=4)

    expected_df = label_info.copy()
    expected_df['phenotype'] =['label2', 'label1', 'label3', 'label1']

    assert_frame_equal(expected_df, returned_df)

def test_compute_count_in_region():
    sample_data = pd.DataFrame()
    sample_data['samples'] = ['s1','s1','s2','s4']
    sample_data['start'] = [0, 4, -1, 7]
    sample_data['stop'] = [1, 6, 4, 9]

    returned_df = ReVeaL._compute_count_in_region(sample_data, 0, 5)
    expected_df = pd.DataFrame()
    expected_df['samples'] = ['s1','s2']
    expected_df['start'] = [2, 1]
    expected_df['stop'] = [2, 1]

    assert_frame_equal(expected_df, returned_df)

def test_compute_mutational_load():

    sample_data = pd.DataFrame()
    sample_data['samples'] = ['s1','s1','s2','s4']
    sample_data['start'] = [0, 4, -1, 7]
    sample_data['stop'] = [1, 6, 4, 9]
    
    samples = sample_data['samples'].unique()

    regions_size = pd.DataFrame()
    regions_size['start'] = [0, 7]
    regions_size['stop'] = [5, 10]

    returned_df = ReVeaL.compute_mutational_load(sample_data, samples.tolist(), regions_size, 'C')

    expected_df = pd.DataFrame()
    expected_df['window']= [('C', 0, 5, -1),('C', 7, 10, -1),  ('C', 0, 5, -1),  ('C', 7, 10, -1), ('C', 0, 5, -1),  ('C', 7, 10, -1)]
    expected_df['count'] = [2, 0, 1, 0, 0, 1]
    expected_df['samples'] = ['s1', 's1', 's2', 's2', 's4', 's4']
    
    assert_frame_equal(expected_df.pivot(index='samples', columns='window',  value='count').fillna(0), returned_df)

    returned_df = ReVeaL.compute_mutational_load(sample_data, samples.tolist(), regions_size, 'C', window_size=2)

    expected_df = pd.DataFrame()
    expected_df['window']= [('C', 0, 5, 0), ('C', 0, 5, 0), ('C', 0, 5, 0), ('C', 2, 5, 1), ('C', 2, 5, 1), ('C', 2, 5, 1), ('C', 4, 5, 2), ('C', 4, 5, 2), ('C', 4, 5, 2), ('C', 7, 10, 0), ('C', 7, 10, 0), ('C', 7, 10, 0), ('C', 9, 10, 1), ('C', 9, 10, 1) , ('C', 9, 10, 1) ]
    expected_df['count'] = [1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
    expected_df['samples'] = ['s1','s2','s4', 's1','s2','s4', 's1','s2','s4', 's1','s2','s4', 's1','s2','s4']

    assert_frame_equal(expected_df.pivot(index='samples', columns='window', value='count').fillna(0), returned_df)



    
