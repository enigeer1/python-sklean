from sklearn.preprocessing import LabelEncoder
import pandas as pd

data = pd.read_csv(r'Narrativedata.csv', index_col=0)
# 要输的是标签,不是特征矩阵，所以允许一维
# 最后一列
y = data.iloc[:, -1]
print(y)
le = LabelEncoder() # 实例化
le = le.fit(y) # 导入数据
label = le.transform(y) # transform接口调取结果
print(label)
print(le.classes_)  # 属性.classes_查看标签中究竟有多少类别
# 也可使用 fit_transform 一步到位
data.iloc[:, -1] = LabelEncoder().fit_transform(data.iloc[:, -1])
print(data)