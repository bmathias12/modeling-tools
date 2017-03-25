# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
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