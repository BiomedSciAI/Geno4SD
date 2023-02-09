# -*- coding: utf-8 -*-
# Copyright 2021 Filippo Utro.  All Rights Reserved.
#

"""
Test cases for Delta funtions

Test cases can be run with the following:
  nosetests -v --with-spec --spec-color
  coverage report -m
"""
import os
import pandas as pd
import logging
from geno4sd.evolution.Delta import delta 
from unittest import TestCase
from pandas._testing import assert_frame_equal

logging.disable(logging.CRITICAL)

def test_calculate_patient_delta_predefined():
    df = pd.DataFrame()
    df['samples']=['sample1','sample2']
    df['ft1'] = [4, 3]
    df['ft2'] = [3, 0]
    df = df.set_index('samples')

    pair_df = pd.DataFrame()
    pair_df[0] = ['sample1']
    pair_df[1] = ['sample2']

    delta_out =  delta.calculate_patient_delta_predefined( df, pair_df )

    assert_frame_equal(delta_out, pd.DataFrame({'ft1':[-1],'ft2':[-3]}, index=['sample1:sample2']))
    
def test_calculate_patient_delta_sliding_window():

    df = pd.DataFrame()
    df['samples']=['sample1','sample2']
    df['ft1'] = [4, 3]
    df['ft2'] = [3, 0]
    df = df.set_index('samples')

    return True
    #assert_frame_equal(None,None)


