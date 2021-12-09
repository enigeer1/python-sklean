from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

X, y = make_blobs(n_samples=500, n_features=2, centers=4, random_state=1)

n_clusters = 3
cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
# 重要属性labels_, 查看聚好的类别，每个样本所对应的类
y_pred = cluster.labels_
