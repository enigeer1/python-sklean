from sklearn.preprocessing import Binarizer
import pandas as pd
from sklearn.impute import SimpleImputer

data = pd.read_csv(r'Narrativedata.csv', index_col=0)
Age = data.loc[:, 'Age'].values.reshape(-1, 1)
imp_mean = SimpleImputer().fit_transform(Age)  # 实例化  默认均值填补


data.loc[:, 'Age'] = imp_mean
# 将年龄二值化

data_2 = data.copy()

X = data_2.iloc[:, 0].values.reshape(-1, 1)
transformer = Binarizer(threshold=15).fit_transform(X)
print(transformer)
data.loc[:, 'Age'] = transformer
print(data_2)

