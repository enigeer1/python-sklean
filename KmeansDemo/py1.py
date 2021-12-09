from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 创建自己的数据集
# make_blobs函数是为了聚类产生数据集
# n_features : 设置每个样本有几个特征值，默认值是2
    # n_samples : 取多少个样本数，默认值100
    # centers : 样本中心点有几个，默认值是3
    # random_state : 设置随机数种子，防止每次生成的数据都修改。默认是np.random。
    # cluster_std : 每个类别的方差。默认值是1.0   # shuffle : 洗乱，默认值是True
    # center_box : 中心确认之后的数据边界，默认值（-10.0， 10.0）

X, y = make_blobs(n_samples=500, n_features=2, centers=4, random_state=1)
print(X)
fig, ax1 = plt.subplots(1)  # 1代表子图的数量
# print(fig, ax1)
# scatter函数画散点图
ax1.scatter(X[:, 0], X[:, 1]
            , marker='o' # 点的形状
            , s=8
            ) # 点的大小
plt.show()
# 如果我们想要看见这个点的分布
color = ["red", 'pink', 'orange', 'gray']
fig, ax1 = plt.subplots(1)  # 1代表子图的数量
for i in range(4):
    # print(i)
    ax1.scatter(X[y == i, 0], X[y == i, 1]
                , marker='o'
                , s=8
                , c=color[i]
                )
plt.show()

from sklearn.cluster import KMeans
n_clusters = 3
cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
# 重要属性labels_, 查看聚好的类别，每个样本所对应的类
y_pred = cluster.labels_
print(y_pred)
# 重要属性cluster_centers_, 查看质心
centroid = cluster.cluster_centers_
print(centroid)
# 重要的属性inertia_,查看总距离平方和
print(cluster.inertia_)
for i in range(n_clusters):
    ax1.scatter(X[y_pred == i, 0], X[y_pred == i, 1]
                , marker="o"
                , s=15
                , c=color[i]
                )
ax1.scatter(centroid[:, 0], centroid[:, 1]
            , marker="x"
            , s=15
            , c="black"
            )
plt.show()
n_clusters = 4
cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
# 重要属性labels_, 查看聚好的类别，每个样本所对应的类
inertia_ = cluster.inertia_
print(inertia_)

n_clusters = 5
cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
# 重要属性labels_, 查看聚好的类别，每个样本所对应的类
inertia_ = cluster.inertia_
print(inertia_)

n_clusters = 6
cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
# 重要属性labels_, 查看聚好的类别，每个样本所对应的类
inertia_ = cluster.inertia_
print(inertia_)

print(X[y_pred == i, 0])
print(X[y_pred == i, 0].shape)
print(X[y_pred == i, 1].shape)
print(X.shape)