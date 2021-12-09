from sklearn.preprocessing import StandardScaler

data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
scaler = StandardScaler() # 实例化
scaler.fit(data)
scaler.mean_
scaler.var_
x_std = scaler.transform(data)
print(x_std)
print(x_std.mean())
print(x_std.std())

scaler.fit_transform(data) # 一步到位


