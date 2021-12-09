from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np

wine = load_wine()
print(np.array(wine).shape)
# 数据
print(wine.data)
# 标签
print(wine.target)
# 特征标题
print(wine.feature_names)
# 标签名
print(wine.target_names)
#
Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data, wine.target, test_size=0.3)
# 建模三部曲
# 1，实例化
clf = tree.DecisionTreeClassifier(criterion="entropy")
# 2，训练
clf = clf.fit(Xtrain, Ytrain)
# 3，导入测试集
score = clf.score(Xtest, Ytest) # 返回预测的准确度
print(score)
feature_names = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']


# 画图
import graphviz
dot_data = tree.export_graphviz(clf
                                , feature_names=feature_names
                                , class_names=['class_0', 'class_1', 'class_2']
                                # ,filled=True
                                # ,rounded=True
                                )
graph = graphviz.Source(dot_data)
print(graph)