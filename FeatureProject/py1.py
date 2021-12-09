from sklearn.preprocessing import MinMaxScaler

data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]

import pandas as pd
pd.DataFrame(data)
# 实现归一化
scaler = MinMaxScaler()   # 实例化
scaler = scaler.fit(data) # fit, 在这里本质是min(x)和max(x)

result = scaler.transform(data)  # 通过接口导出结果

result_ = scaler.fit_transform(data)

print(result_)

# 还原归一化的结果
inverse_result = scaler.inverse_transform(result_)
print(inverse_result)

# 使用MinMaxScaler的参数feature_range实现将数据归到[0, 1]以外的范围
data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
scaler1 = MinMaxScaler(feature_range=[5, 10])
result = scaler1.fit_transform(data)
print(result)
