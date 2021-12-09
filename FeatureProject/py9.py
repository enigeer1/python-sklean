from sklearn.feature_selection import VarianceThreshold
import pandas as pd

X = pd.read_csv(r"digit recognizor.csv")
print(X.shape)
selector = VarianceThreshold() # 实例化，不填参数方差默认为0

X_var0 = selector.fit_transform(X)
print(X_var0.shape)

import numpy as np
# var():取每列的方差
print(np.median(X.var().values))
x_fsvar = VarianceThreshold(np.median(X.var().values)).fit_transform(X)
print(x_fsvar.shape)
# 若特征是伯努利随机分布，假设片0.8，即二分类特征中占到80%以上的时候删除特征
X_bvar = VarianceThreshold(.8 * .2).fit_transform(X)
print(X_bvar.shape)
