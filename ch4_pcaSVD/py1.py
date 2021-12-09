import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# 2, 提取数据集
iris = load_iris()
y = iris.target
x = iris.data
print("y:", y)
print(x.shape)

import pandas as pd
pd.DataFrame(x)
# 3, 建模
pca = PCA(n_components=2) # 实例化
pca = pca.fit(x) # 拟合模型
x_dr = pca.transform(x) # 获取新矩阵
# print("x_dr[y == 0, 0], ", x_dr[y == 0, 0])
# print(x_dr)
# 4, 可视化
colors = ['red', 'black', 'orange']
print(iris.target_names)
plt.figure()
for i in [0, 1, 2]:
    plt.scatter(x_dr[y == i, 0]
                , x_dr[y == i, 1]
                , alpha=.7 # 点的透明度
                , c=colors[i]
                , label=iris.target_names[i])
plt.legend()
plt.title("PCA of IRIS datasets")
plt.show()

# 6 探索降维后的数据
# explained_variance_查看每个特征向量上所带的信息量(可解释性方差的大小)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)



