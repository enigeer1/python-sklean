from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#  加载数据
data = load_breast_cancer()

#
rfc = RandomForestClassifier(n_estimators=100, random_state=90) # 实例化
score_pre = cross_val_score(rfc, data.data, data.target, cv=10).mean()
print(score_pre)

scorel = []
for i in range(0, 200, 10):
    rfc = RandomForestClassifier(n_estimators=i+1,
                                 n_jobs=-1,
                                 random_state=90)
    score = cross_val_score(rfc, data.data, data.target, cv=10).mean()
    scorel.append(score)
print(max(scorel), (scorel.index(max(scorel))*10)+1)
plt.figure(figsize=[20, 5])
plt.plot(range(1, 201, 10), scorel)
plt.show()


scorel = []
for i in range(25, 40):
    rfc = RandomForestClassifier(n_estimators=i,
                                 n_jobs=-1,
                                 random_state=90)
    score = cross_val_score(rfc, data.data, data.target, cv=10).mean()
    scorel.append(score)
print(max(scorel), ([*range(25, 40)][scorel.index(max(scorel))]))
plt.figure(figsize=[20, 5])
plt.plot(range(25, 40), scorel)
plt.show()

# 6, 为网格搜索做准备， 书写网格搜索的参数
# 7, 开始按照参数对模型整体准确率的影响程度进行体调参，首先调整max_depth
# 调整max_depth
param_grid = {'max_depth': np.arange(1, 20, 1)}
rfc = RandomForestClassifier(n_estimators=26
                             , random_state=90
                             )
GS = GridSearchCV(rfc, param_grid, cv=10)
GS.fit(data.data, data.target)
print(GS.best_params_)
print(GS.best_score_)

# 8，调整max_features
# max_features是唯一一个即能够将模型往左（低方差高偏差）推，
# 也能够将模型往右（高方差低偏差）
param_grid = {'max_features': np.arange(5, 30, 1)}

rfc = RandomForestClassifier(n_estimators=26
                             , random_state=90)
GS = GridSearchCV(rfc, param_grid, cv=10)
GS.fit(data.data, data.target)

print(GS.best_params_)
print(GS.best_score_)

# 9，调整min_samples_leaf
param_grid = {'min_samples_leaf': np.arange(1, 11, 1)}

rfc = RandomForestClassifier(n_estimators=26
                             , random_state=90)
GS = GridSearchCV(rfc, param_grid, cv=10)
GS.fit(data.data, data.target)

print(GS.best_params_)
print(GS.best_score_)

# 10, 继续尝试min_samples_split
param_grid = {'min_samples_split': np.arange(1, 11, 1)}

rfc = RandomForestClassifier(n_estimators=26
                             , random_state=90)
GS = GridSearchCV(rfc, param_grid, cv=10)
GS.fit(data.data, data.target)
print(GS.best_params_)
print(GS.best_score_)
# 11. 最后尝试一下criterion

param_grid = {'criterion': ['gini', 'entropy']}
rfc = RandomForestClassifier(n_estimators=26
                             , random_state=90)
GS = GridSearchCV(rfc, param_grid, cv=10)
GS.fit(data.data, data.target)
print(GS.best_params_)
print(GS.best_score_)
# 12, 调整完毕，总结出模型的最佳参数
rfc = RandomForestClassifier( n_estimators=26
                             , random_state=90
                             , min_impurity_split=3
                             , max_depth=9
                             , max_features=6
                             , max_leaf_nodes=1)