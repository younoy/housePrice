import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV

train_df = pd.read_csv("houseData/train.csv",index_col=0)
test_df = pd.read_csv("houseData/test.csv",index_col=0)
# 合并处理,先拿出 SalePrice
#  log1p 使处理的数据更加柔滑，更加正态
price = pd.DataFrame({"price":train_df['SalePrice'],"log(price+1)":np.log1p(train_df["SalePrice"])})
price.hist()
y_train = np.log1p(train_df.pop("SalePrice"))

'''
# 画直方图
fig = plt.figure()
price['price'].plot(kind='bar',stacked=True)
plt.show()
'''

all_Data = pd.concat((train_df,test_df),axis=0)
# print(all_Data.info())

'''
    特征工程：
        字符串： 1,one-hot 表示出字母类别化，也就是数字化
        数字型： 2,处理缺失值,采用平均值
        3, 标准化数字型数据，变得更加平滑
'''
all_Data['MSSubClass'] = all_Data['MSSubClass'].astype(str)
# print(all_Data['MSSubClass'].value_counts())
# MSSubClass_df = pd.get_dummies(all_Data['MSSubClass'],prefix='MSSubClass')
all_dump_data = pd.get_dummies(all_Data)

# print(all_dump_data.isnull().sum().sort_values(ascending=False).head())
mean_cols = all_dump_data.mean()
all_dump_data = all_dump_data.fillna(mean_cols)
# print(all_dump_data.isnull().sum().sum())

numeric_cols = all_Data.columns[all_Data.dtypes != "object"]
# print(numeric_cols)
all_data_mean = all_dump_data.loc[:,numeric_cols].mean()
all_data_std = all_dump_data.loc[:,numeric_cols].std()
all_dump_data.loc[:,numeric_cols] = (all_dump_data.loc[:,numeric_cols] - all_data_mean)/all_data_std

# print(all_dump_data)

'''
    模型选择：
        回归模型：
            1,ridge
        优化：
            bagging+boost
            xgboost
'''

train_x = all_dump_data.loc[train_df.index].values
train_y = y_train.values
test_x = all_dump_data.loc[test_df.index].values

from sklearn.linear_model import Ridge

# model = Ridge(alpha = 0.5)
# model.fit(train_x,train_y)
# print(model.score(train_x,train_y))

params = {'alpha':[1,10,15,20,25,30]}
ridge = Ridge()


gsearch = GridSearchCV(ridge,param_grid=params,cv=10)
gsearch.fit(train_x,train_y)
# print(gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_)
result_y = pd.DataFrame(np.expm1(gsearch.predict(test_x)),columns=['SalePrice'])
# print(result_y)
result_id = pd.DataFrame(test_df.index)
# print(len(result_id),len(result_y))
result = pd.concat((result_id,result_y),axis=1)
result.to_csv('sample_submission.csv',index=False)

''' 
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import cross_val_score
testScores = []
for param in params:
    clf = BaggingRegressor(n_estimators=param,base_estimator=ridge)
    testScore = np.sqrt(-cross_val_score(clf,train_x,train_y,cv=10,scoring="neg_mean_squared_error"))
    testScores.append(np.mean(testScore))

import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(params,testScores)
plt.title("CV Error")
plt.show()
'''
