# %matplotlib inline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine


wine = load_wine()
print(wine)
print(wine.data.shape)

# 实例化
# 训练集带入实例化后的模型去进行训练，使用的接口是fit
# 使用其他接口将测试集导入训练好的模型， 去获取我们获取结果（score, Y_test）
from sklearn.model_selection import train_test_split
# 划分测试集与训练集
Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data, wine.target, test_size=0.3)
clf = DecisionTreeClassifier(random_state=0)
rfc = RandomForestClassifier(random_state=0)

clf = clf.fit(Xtrain, Ytrain)
rfc = rfc.fit(Xtrain, Ytrain)

score_c = clf.score(Xtest, Ytest)
score_r = rfc.score(Xtest, Ytest)

print("Single Tree:{}".format(score_c)
      , "Random Forest:{}".format(score_r))

# 交叉验证: cross_val_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

rfc = RandomForestClassifier(n_estimators=10)
rfc_s = cross_val_score(rfc, wine.data, wine.target, cv=10)

clf = DecisionTreeClassifier()
clf_s = cross_val_score(clf, wine.data, wine.target, cv=10)

plt.plot(range(1, 11), rfc_s, label="RandomForest")
plt.plot(range(1, 11), clf_s, label="DecisionTree")
plt.legend()
plt.show()


# 画出随机森林和决策树在十组交叉验证下的效果对比
rfc_l = []
clf_l = []
for i in range(10):
    rfc = RandomForestClassifier(n_estimators=25)
    rfc_s = cross_val_score(rfc, wine.data, wine.target, cv=10).mean()
    rfc_l.append(rfc_s)

    clf = DecisionTreeClassifier()
    clf_s = cross_val_score(clf, wine.data, wine.target, cv=10).mean()
    clf_l.append(clf_s)
plt.plot(range(1, 11), rfc_l, label = "Random Forest")
plt.plot(range(1, 11), clf_l, label = "Decision Tree")
plt.legend()
plt.show()



for i in range(len(rfc.estimators_)):
    print(rfc.estimators_[i].random_state)