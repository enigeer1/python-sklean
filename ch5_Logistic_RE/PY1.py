from sklearn.linear_model import LogisticRegression as LR
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
x = data.data
y = data.target
print(x.shape)
print(data.data.shape)

lrl1 = LR(penalty="l1", solver="liblinear", C=0.5, max_iter=1000)
lrl2 = LR(penalty="l2", solver="liblinear", C=0.5, max_iter=1000)

# 逻辑回归的重要coef_, 查看每个特征所对应的参数
lrl1 = lrl1.fit(x, y)
print(lrl1.coef_)
lrl2 = lrl2.fit(x, y)
print(lrl2.coef_)
print("*"*90)
l1 = []
l2 = []
l1test = []
l2test = []
xtrain,  xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=420)
# np.linspace(0.05, 1, 19) 从0.05开始，到1，取19个数字
#
for i in np.linspace(0.05, 1, 19):
    lrl1 = LR(penalty="l1", solver="liblinear", C=i, max_iter=1000)
    lrl2 = LR(penalty="l2", solver="liblinear", C=i, max_iter=1000)

    lrl1 = lrl1.fit(xtrain, ytrain)
    l1.append(accuracy_score(lrl1.predict(xtrain), ytrain))
    l1test.append(accuracy_score(lrl1.predict(xtest), ytest))

    lrl2 = lrl2.fit(xtrain, ytrain)
    l2.append(accuracy_score(lrl2.predict(xtrain), ytrain))
    l2test.append(accuracy_score(lrl2.predict(xtest), ytest))

graph = [l1, l2, l1test, l2test]
color = ['green', 'black', 'lightgreen', 'gray']
label = ['L1', 'L2', 'L1test', 'L2test']

plt.figure(figsize=(6, 6)) # 6*6的画布
for i in range(len(graph)):
    plt.plot(np.linspace(0.05, 1, 19), graph[i], color[i], label=label[i])
plt.legend(loc=4) # 图例的显示位置
plt.show()

