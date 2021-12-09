# 相关性过滤--卡方过滤
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# 假设一直需要300个特征

x_fschi = SelectKBest(chi2)