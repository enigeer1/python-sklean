from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd

data = pd.read_csv(r"digit recognizor.csv")
x = data.iloc[:, 1:]
y = data.iloc[:, 0]

# VarianceThreshold 代表方差过滤
x_fsvar = VarianceThreshold(np.median(x.var().values)).fit_transform(x)
print(x_fsvar.shape)
# %%timeit
# 2, KNN()方差过滤前
print("KNN()方差过滤前:", cross_val_score(KNN(), x, y, cv=5).mean())

print("KNN()方差过滤后:", cross_val_score(KNN(), x_fsvar, y, cv=5).mean())
# 3, 随机森林 - 方差过滤前
print("随机森林 - 方差过滤前:", cross_val_score(RFC(n_estimators=10, random_state=0), x, y, cv=5).mean())
print("随机森林 - 方差过滤后:", cross_val_score(RFC(n_estimators=10, random_state=0), x_fsvar, y, cv=5).mean())

