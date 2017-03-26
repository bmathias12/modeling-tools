# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def get_bins(scores, q=10, adjust_endpoints=False):
    _, bins =  pd.qcut(scores, q=q, retbins=True)
    if adjust_endpoints:
        bins[0] = 0.0
        bins[len(bins)-1] = 1.0
    return bins

def get_quantiles_from_bins(scores, bins, one_high=True):
    quantiles = pd.Series(np.digitize(scores, bins, right=True))
    if one_high:
        q = len(bins)-1
        quantiles = q - quantiles + 1
    return quantiles
    
    