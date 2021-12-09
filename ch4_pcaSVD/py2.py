import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import numpy as np

# 2, 提取数据集
iris = load_iris()
y = iris.target
x = iris.data

pca_line = PCA().fit(x)
plt.plot([1, 2, 3, 4], np.cumsum(pca_line.explained_variance_ratio_))
plt.xticks([1, 2, 3, 4])  # 这是为了限制坐标显示为整数
plt.xlabel("number of component after dimension reduction")
plt.ylabel("cumulative explained variance")
plt.show()
#
