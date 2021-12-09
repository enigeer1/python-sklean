from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
# OrdinalEncoder: 特征专用，能够将分类特征转换为分类数值
data = pd.read_csv(r'Narrativedata.csv', index_col=0)

data_ = data.copy()
print(data_.head())
# categories_属性
# print(OrdinalEncoder().fit(data_.iloc[:, 1: 2]).categories_)

data_.iloc[:, 1: 2] = OrdinalEncoder().fit_transform(data_.iloc[:, 1: 2])
print(data_.head())

from sklearn.preprocessing import OneHotEncoder
X = data.iloc[:, 1: -1]
enc = OneHotEncoder(categories='auto').fit(X)
result = enc.transform(X).toarray()
print(result.shape)
print(enc.get_feature_names())
# axis=1表示跨行合并，就是量表左右相连，
newdata = pd.concat([data, pd.DataFrame(result)], axis=1)
print(newdata.head())
# axis=1 表示列 inplace=True 覆盖原数据
newdata.drop(["Sex", "Embarked"], axis=1, inplace=True)
print("newdata", newdata)