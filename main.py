import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

train = pd.read_csv("./FIFA_train.csv")
test = pd.read_csv("./FIFA_test.csv")
submission = pd.read_csv("./submission.csv")

train.drop(['id', 'name'], axis=1, inplace=True)
test.drop(['id', 'name'], axis=1, inplace=True)

def func(string:object):
    string = string[-4:] # 뒤에서 네번째 까지(연도))
    return int(string)

train['contract_until'] = train['contract_until'].apply(func)
test['contract_until'] = test['contract_until'].apply(func)

label_e = LabelEncoder()
train['continent'] = label_e.fit_transform(train['continent'])
train['position'] = label_e.fit_transform(train['position'])
train['prefer_foot'] = label_e.fit_transform(train['prefer_foot'])
test['continent'] = label_e.fit_transform(test['continent'])
test['position'] = label_e.fit_transform(test['position'])
test['prefer_foot'] = label_e.fit_transform(test['prefer_foot'])

sdscaler = StandardScaler()
sdscaled_train = pd.DataFrame(sdscaler.fit_transform(train), columns = train.columns)
sdscaled_test = pd.DataFrame(sdscaler.fit_transform(test), columns = test.columns)

X_train, X_test, y_train, y_test = train_test_split(train, train['value'],test_size=0.2,random_state=2022)

# 단일 결정 트리
dt_regr = DecisionTreeRegressor(max_depth=4)
dt_regr.fit(X_train['stat_overall'].values.reshape(-1,1), y_train)
y_pred = dt_regr.predict(X_test['stat_overall'].values.reshape(-1,1))
print('단순 결정 트리 회귀 R2 : {:.4f}'.format(r2_score(y_test, y_pred))) # depth = 5

arr = np.arange(1,10)
best_depth = 0
best_r2 = 0

for depth in arr:
    dt_regr = DecisionTreeRegressor(max_depth=depth)
    dt_regr.fit(X_train['stat_overall'].values.reshape(-1, 1), y_train)
    y_pred = dt_regr.predict(X_test['stat_overall'].values.reshape(-1, 1))

    temp_r2 = r2_score(y_test, y_pred)
    print('\n단순 결정 트리 회귀 depth ={} R2 : {:.4f}'.format(depth, temp_r2))

    if best_r2 < temp_r2:
        best_depth = depth
        best_r2 = temp_r2

print('최적의 결과는 depth={} r2={:.4f}'.format(best_depth, best_r2))


# 다중 결정 트리
dt_regr = DecisionTreeRegressor(max_depth=5)
dt_regr.fit(X_train, y_train)
y_pred = dt_regr.predict(X_test)
print('다중 결정 트리 회귀 R2 : {:.4f}'.format(r2_score(y_test, y_pred)))

best_depth = 0
best_r2 = 0

for depth in arr:
    dt_regr = DecisionTreeRegressor(max_depth=depth)
    dt_regr.fit(X_train, y_train)
    y_pred = dt_regr.predict(X_test)

    temp_r2 = r2_score(y_test, y_pred)
    print('\n다중 결정 트리 회귀 depth ={} R2 : {:.4f}'.format(depth, temp_r2))

    if best_r2 < temp_r2:
        best_depth = depth
        best_r2 = temp_r2

print('최적의 결과는 depth={} r2={:.4f}'.format(best_depth, best_r2))
