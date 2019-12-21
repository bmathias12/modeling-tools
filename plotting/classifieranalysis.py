import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

class ClassiferAnalysis:
    def __init__(self, y_true, scores):
        self.y_true = y_true
        self.scores = scores

    def plot_roc_curve(self, figsize=(8,5)):
        fpr, tpr, _ = metrics.roc_curve(self.y_true, self.scores)
        auc = metrics.roc_auc_score(self.y_true, self.scores)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(fpr, tpr, label='Area Under Curve: {:0.3f}'.format(auc))
        ax.plot([0,1], [0,1], color='black', linestyle='--')
        plt.title(title, fontweight='bold')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')

    def plot_ks(self):
        raise NotImplementedError

    def plot_prediction_density(self):
        raise NotImplementedError

    def plot_quantiles(self, q=10):
        raise NotImplementedError

    def gains_table(self):
        raise NotImplementedError
