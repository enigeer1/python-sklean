from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt
# 实例化
regressor = DecisionTreeRegressor(random_state=0)

# numpy.random.RandomState()是一个伪随机数生成器, 此命令将会产生一个随机状态种子,
# 在该状态下生成的随机序列（正态分布）一定会有相同的模式
rng = np.random.RandomState(1)
print(rng.rand(16))
X = np.sort(5 * rng.rand(80, 1), axis=0)
Y = np.sin(X).ravel()
# 画图
plt.figure()
plt.scatter(X, Y, s=20, edgecolors="black", c="darkorange", label="data")

Y[::5] += 3 * (0.5 - rng.rand(16))

regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X, Y)
regr_2.fit(X, Y)
# 测试集导入模型，预测结果
X_test = np.arange(0, 5, 0.01)[:, np.newaxis]

y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)
# 画图
plt.figure()
plt.scatter(X, Y, s=20, edgecolors="black", c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Three Regression")
plt.legend()
plt.show()
