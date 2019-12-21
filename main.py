
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

RANDOM_STATE = 123


def make_model():
    X, y = make_classification(
        n_samples=10000,
        n_features=30,
        n_informative=10,
        n_redundant=3,
        n_repeated=2,
        class_sep=1.0,
        random_state=RANDOM_STATE
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=RANDOM_STATE
        )

    clf = LogisticRegression(random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:,1]

    return y_test, probs

y_true, scores = make_model()

from plotting.classifieranalysis import ClassiferAnalysis

ca = ClassiferAnalysis(y_true, scores)

print(ca.gains_table())