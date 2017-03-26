
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from modutils import plotting
from modutils import quantiles

RANDOM_STATE = 123

X, y = make_classification(
    n_samples=10000,
    n_features=30,
    n_informative=10,
    n_redundant=3,
    n_repeated=2,
    class_sep=0.5,
    random_state=RANDOM_STATE
    )

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=RANDOM_STATE
    )

clf = LogisticRegression(random_state=RANDOM_STATE)
clf.fit(X_train, y_train)

preds = clf.predict(X_test)
probs = clf.predict_proba(X_test)[:,1]

# Check Plotting Functions
plotting.plot_roc_curve(y_test, probs)
plotting.plot_prediction_density(y_test, probs)

# Check quantile functions
bins = quantiles.get_bins(probs, q=10, adjust_endpoints=True)
deciles = quantiles.get_quantiles_from_bins(probs, bins)

# Check that mean by decile is descending from 1 to 10
d = pd.concat([pd.Series(probs), deciles], axis=1)
d.columns = ['scores', 'deciles']
d.groupby('deciles')['scores'].mean()
