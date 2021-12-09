import pandas as pd

data = pd.read_csv(r'Narrativedata.csv', index_col=0)
print(data.head())
print(data.info())
# loc[:, 'Age']:取
# reshape(-1, 1) 表示升维
Age = data.loc[:, 'Age'].values.reshape(-1, 1)
# print(Age[:20])

from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer()  #实例化  默认均值填补
imp_median = SimpleImputer(strategy="median")  # 用中位数填补
img_0 = SimpleImputer(strategy="constant", fill_value=0)  # 用0填补

# img_mean = imp_mean.fit_transform(Age)
# print("img_mean:", img_mean)
# imp_median = imp_median.fit_transform(Age)
# print("imp_median:", imp_median)
# img_0 = img_0.fit_transform(Age)
# print("img_0:", img_0)
data.loc[:, 'Age'] = imp_median
print(data.info())
# 使用众数填补Embarked
Embarked = data.loc[:, 'Embarked'].values.reshape(-1, 1)
imp_most = SimpleImputer(strategy='most_frequent')
data.loc[:, 'Embarked'] = imp_most
print(data.info())



