# encoding: utf-8


"""
@author: PDXG
@file: random_forest_upsample.py
@create_time: 2019/11/27 19:56
@version: 
@description: 
"""

# %%

# 导包
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# %%

# 读取数据
# 根据题意，先把header设置好
headers = ['age', 'workclass', 'fnlwgt',
           'education', 'education_num', 'marital_status',
           'occupation', 'relationship', 'race',
           'sex', 'capital_gain', 'capital_loss',
           'hours_per_week', 'native_country', 'income']

# 读取训练集进入内存
train_data = pd.read_csv("data/adult_train.csv", names=headers)

# 对于测试集，他的第一行是一个分隔，不读入
test_data = pd.read_csv("data/adult_test.csv", names=headers, skiprows=[0])

# %%

# 看看训练集有多少数据
train_data.shape

# %%

# 看看数据是怎么样的，平均值；最大最小值，标准差等等（只有连续性的）
train_data.describe()

# %%

# 数据中有一些值是？，用python的NaN代替方便以后直接当作空值处理
train_data = train_data.replace('[?]', np.NaN, regex=True)
test_data = test_data.replace('[?]', np.NaN, regex=True)
# 看看数据里有NaN的各类有多少
train_data.isnull().sum()
test_data.isnull().sum()

# %%

# 对于缺失属性的数据直接删除
train_data = train_data.dropna()
train_data.isnull().sum()
test_data = test_data.dropna()
test_data.isnull().sum()

# %%

# 开始处理离散数据
# income我们需要将其映射一下
# 先将>=50K的映射成0，<=50K的设置为1
income_map = {' <=50K': 1, ' >50K': 0}
income_map2 = {' <=50K.': 1, ' >50K.': 0}
train_data['income'] = train_data['income'].map(income_map).astype(int)
test_data['income'] = test_data['income'].map(income_map2).astype(int)

# %%

# 取出所有的离散量属性
discrete = [x for x in train_data.columns if train_data[x].dtype == 'object']
discrete

# %%

# 看看这些离散量属性的取值分布
for i in discrete:
    print(train_data[i].value_counts())

# %%

# 发现有一些是比较接近重叠的，将这些重叠的归到同一类
# 大致可以把gov有关的一类，Private一类，Self-emp一类，没工作的一类
train_data['workclass'] = train_data['workclass'].replace([' Self-emp-not-inc', ' Self-emp-inc'], ' Self-emp')
train_data['workclass'] = train_data['workclass'].replace([' Federal-gov', ' Local-gov', ' State-gov'], ' Gov')
train_data['workclass'] = train_data['workclass'].replace([' Without-pay', ' Never-worked'], ' Un-emp')
train_data['workclass'].value_counts()

test_data['workclass'] = test_data['workclass'].replace([' Self-emp-not-inc', ' Self-emp-inc'], ' Self-emp')
test_data['workclass'] = test_data['workclass'].replace([' Federal-gov', ' Local-gov', ' State-gov'], ' Gov')
test_data['workclass'] = test_data['workclass'].replace([' Without-pay', ' Never-worked'], ' Un-emp')

# %%

# 同理对marital_status进行归类
train_data['marital_status'] = train_data['marital_status'].replace(
    [' Divorced', ' Married-spouse-absent', ' Never-married', ' Separated', ' Widowed'], ' Single')
train_data['marital_status'] = train_data['marital_status'].replace([' Married-civ-spouse', ' Married-AF-spouse'],
                                                                    ' Couple')
train_data['marital_status'].value_counts()

test_data['marital_status'] = test_data['marital_status'].replace(
    [' Divorced', ' Married-spouse-absent', ' Never-married', ' Separated', ' Widowed'], ' Single')
test_data['marital_status'] = test_data['marital_status'].replace([' Married-civ-spouse', ' Married-AF-spouse'],
                                                                  ' Couple')

# %%

# native_country这个分类太多了，而且很多类的人很少，干脆把人少的都归到other里
train_data['native_country'] = train_data['native_country'].replace([' Holand-Netherlands', ' Scotland', ' Honduras',
                                                                     ' Hungary', ' Outlying-US(Guam-USVI-etc)',
                                                                     ' Yugoslavia',
                                                                     ' Laos', ' Thailand', ' Cambodia',
                                                                     ' Trinadad&Tobago', ' Hong', ' Ireland',
                                                                     ' France', ' Ecuador', ' Greece', ' Peru',
                                                                     ' Nicaragua', ' Portugal', ' Iran',
                                                                     ' Taiwan', ' Haiti'], ' Other')

test_data['native_country'] = test_data['native_country'].replace([' Holand-Netherlands', ' Scotland', ' Honduras',
                                                                   ' Hungary', ' Outlying-US(Guam-USVI-etc)',
                                                                   ' Yugoslavia',
                                                                   ' Laos', ' Thailand', ' Cambodia',
                                                                   ' Trinadad&Tobago', ' Hong', ' Ireland',
                                                                   ' France', ' Ecuador', ' Greece', ' Peru',
                                                                   ' Nicaragua', ' Portugal', ' Iran',
                                                                   ' Taiwan', ' Haiti'], ' Other')
train_data['native_country'].value_counts()

# %%

# 最后，看看education_num这个连续量
train_data['education'].value_counts()

# %%

# 发现跟education一样的，重复了，因为eudcation是离散的，不好处理。去掉这个属性
train_data = train_data.drop(columns=['education'])
test_data = test_data.drop(columns=['education'])

# %%

# 看看相关系数矩阵,检查一下连续变量。发现序号属性不太影响最后的收入
train_data.corr()

# %%

# 把序号属性删掉
train_data = train_data.drop(columns=['fnlwgt'])
test_data = test_data.drop(columns=['fnlwgt'])

# %%

# 进行one-hot编码
train_data = pd.get_dummies(train_data, columns=['workclass', 'marital_status', 'occupation',
                                                 'relationship', 'race', 'sex',
                                                 'native_country'])
train_data

test_data = pd.get_dummies(test_data, columns=['workclass', 'marital_status', 'occupation',
                                               'relationship', 'race', 'sex',
                                               'native_country'])

# %%

# 看看编码后的结果
train_data.columns

# %%

# 将非2值类型的数据进行标准化
train_data_need_to_standard = train_data[['age', 'education_num', 'capital_gain',
                                          'capital_loss', 'hours_per_week']]
train_data_need_to_standard

# %%

scaler = StandardScaler()
scaler.fit(train_data_need_to_standard)

# %%

train_data_standard = pd.DataFrame(scaler.transform(train_data_need_to_standard))
train_data_standard.head()

# %%

# 将标准化的数据添回原来的整个表里
# 但是我们之前可以看到他的列属性名字没有了，加回去
column_name = ['age', 'education_num', 'capital_gain',
               'capital_loss', 'hours_per_week']
train_data_standard.columns = column_name
train_data_standard

# %%

# 用标准化数据覆盖原来的数据
for i in train_data_standard.columns:
    train_data[i] = train_data_standard[i]
train_data = train_data.dropna()
# train_data

# %%

# 以>50k来分类，样本的分布怎样
x_ = train_data.drop('income', axis=1)
y_ = train_data.income

sns.countplot(train_data['income'], Label='count')

# %%

# 重采样
train_data_less = train_data[train_data.income == 0]
train_data_more = train_data[train_data.income == 1]

train_data.income.value_counts()

# %%
train_data_upsampled = resample(train_data_less,
                                replace=True,
                                n_samples=21024,
                                random_state=123)

# 将重采样的数据添加回去
train_data = pd.concat([train_data_more, train_data_upsampled])

# %%

train_data.income.value_counts()

# %%

# 把要预测的值income单独拿出来
y_train = train_data.income
x_train = train_data.drop('income', axis=1)

y_test = test_data.income
x_test = test_data.drop('income', axis=1)

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

# %%

random_forest = RandomForestClassifier(n_estimators=10000)
random_forest.fit(x_train, y_train)

# %%

y_pred = random_forest.predict(x_test)
print(accuracy_score(y_test, y_pred) * 100)

# %%

random_forest_confusion_matrix = confusion_matrix(y_test, y_pred)
print(random_forest_confusion_matrix)
