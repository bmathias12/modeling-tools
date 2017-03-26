# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def ks_table(y, quantiles):
    """Function to produce a K-S table"""
    if len(set(y)) > 2:
        raise ValueError('Function only defined for binary classification')
    
    df = pd.concat([pd.Series(y), pd.Series(quantiles)], axis=1)
    df.columns = ['y', 'quantiles']
    
    # Get counts of positive and negative values
    count_total = df.groupby('quantiles')['y'].count()
    count_pos = df.groupby('quantiles')['y'].sum()
    count_neg = count_total - count_pos 

    # Get cumulative percents
    pos_pct_cum = count_pos.cumsum() / float(count_pos.sum()) * 100
    neg_pct_cum = count_neg.cumsum() / float(count_neg.sum()) * 100
    
    # Calculate KS
    ks_idx = np.abs(pos_pct_cum - neg_pct_cum)
    
    # Output table
    out_df = pd.concat([
            count_total, count_pos, count_neg, pos_pct_cum, neg_pct_cum, ks_idx
        ], axis=1)
    out_df.columns = [
            'count_total', 'count_pos', 'count_neg', 
            'pos_pct_cum', 'neg_pct_cum', 'ks_idx'
        ]
    
    return out_df
    