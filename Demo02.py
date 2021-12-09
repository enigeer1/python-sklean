from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

boston = load_boston()
# 实例化
# random_state 这里的random_state就是为了保证程序每次运行都分割一样的训练集和测试集。
# 否则，同样的算法模型在不同的训练集和测试集上的效果不一样。
regressor = DecisionTreeRegressor(random_state=0)
# cv是分10类
cross_val_score(regressor, boston.data, boston.target, cv=10)