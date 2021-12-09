import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import numpy as np

# 2, 提取数据集
iris = load_iris()
x = iris.data

pca_mle = PCA(n_components="mle").fit(x)
x_mle = pca_mle.transform(x)

print(x_mle)
# 可以发现，mle为我们自动选了3个特征
print(pca_mle.explained_variance_ratio_.sum())