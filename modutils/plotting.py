# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from sklearn import metrics


def plot_roc_curve(
        y, 
        scores, 
        pos_label=1,
        title='ROC Curve',
        figsize=(8,5),
        saveloc=None):
    """Function to plot a ROC curve when given true values and scores"""
    fpr, tpr, _ = metrics.roc_curve(y, scores, pos_label=pos_label)
    auc = metrics.roc_auc_score(y, scores)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, label='Area Under Curve: {:0.3f}'.format(auc))
    ax.plot([0,1], [0,1], color='black', linestyle='--')
    plt.title(title, fontweight='bold')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    
    if saveloc:
        plt.savefig(saveloc, bbox_inches='tight')
    else:
        plt.show()

def plot_prediction_density(
    y,
    scores,
    title='Prediction Density',
    figsize=(8,5),
    saveloc=None):
    """Function to plot the kernel density for model predictions"""
    
    class_set = sorted(set(y))
    if len(class_set) > 2:
        raise ValueError('Function defined for binary classification only')
    
    x_grid = np.linspace(0, 1, 1000)
    
    fig, ax = plt.subplots(figsize=figsize)
    for v in class_set:
        a = scores[y == v]
        kernel = gaussian_kde(a, bw_method='scott')
        kde = kernel.evaluate(x_grid)
        ax.plot(x_grid, kde, linewidth=2.5, label='Target = {}'.format(v))
        ax.fill_between(x_grid, kde, alpha=0.6)
    plt.title(title, fontweight='semibold')
    plt.legend()
    
    if saveloc:
        plt.savefig(saveloc, bbox_inches='tight')
    else:
        plt.show()
    