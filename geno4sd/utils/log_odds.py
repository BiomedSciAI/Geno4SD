__author__ = "Kahn Rhrissorrakrai"
__copyright__ = "Copyright 2022, IBM Research"
__version__ = "0.0.1"
__maintainer__ = "Kahn Rhrissorrakrai"
__email__ = "krhriss@us.ibm.com"
__status__ = "Development"

import numpy as np
import pandas as pd
from scipy.stats import norm

def calculate(ctgTable):
    """
    Log Odds Test
    
    :param ctgTable: <dataframe> 2x2 contigency table
    :return: <series> Log odds statistics including, odds ratio (odr), log odds ratio (lodr), z-score (z), p-value (pval), 95% confidence interval (conf_int)

    """
    odr = (ctgTable.iloc[0, 0]/ctgTable.iloc[1, 0])/(ctgTable.iloc[0, 1]/ctgTable.iloc[1, 1])
    lodr = np.log10(odr)
    stdvarlodr = (1/ctgTable.iloc[0, 0]) + (1/ctgTable.iloc[1, 0]) + (1/ctgTable.iloc[0, 1]) + (1/ctgTable.iloc[1, 1])
    z = lodr/stdvarlodr
    pval = 2 * (1 - norm.cdf(np.abs(z)))
    confInt_upper = np.exp( np.log(odr) + (1.96 * np.sqrt(stdvarlodr) ) )
    confInt_lower = np.exp( np.log(odr) - (1.96 * np.sqrt(stdvarlodr) ) )
    return( pd.Series( [ odr, lodr, z, pval, str(round(confInt_lower,3))+','+str(round(confInt_upper,3)) ], index = ['odr', 'lodr', 'z', 'pval', 'conf_int' ] ) )