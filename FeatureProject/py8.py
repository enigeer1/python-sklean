from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd

data = pd.read_csv(r'Narrativedata.csv', index_col=0)
x = data.iloc[:, 0].values.reshape(-1, 1)
est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
print(est.fit_transform(x))
# 