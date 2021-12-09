import sklearn.metrics
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

boston = load_boston()

print(boston)
# random_state是一个随机种子，是在任意带有随机性的类或函数里作为参数来控制随机模式
regressor = RandomForestRegressor(n_estimators=100, random_state=0) # 实例化
cross_val_score(regressor, boston.data, boston.target, cv=10
                ,scoring="neg_mean_squared_error")
# sklearn 当中的模型评估指标(打分)列表

print(sorted(sklearn.metrics.SCORERS.keys()))