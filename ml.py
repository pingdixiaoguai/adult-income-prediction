# encoding: utf-8


"""
@author: PDXG
@file: ml.py
@create_time: 2019/11/23 16:55
@version: 
@description: 
"""

import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 根据题意，先把header设置好
    headers = ['age', 'workclass', 'fnlwgt',
               'education', 'education-num', 'marital-status',
               'occupation', 'relationship', 'race',
               'sex', 'capital-gain', 'capital-loss',
               'hours-per-week', 'native-country', 'predclass']

    # 读取数据进入内存,na_values就是将NA/NaN的值替换为指定的“？”
    train_data = pd.read_csv("data/adult_train.csv",
                             header=None,
                             names=headers,
                             na_values=["?"],
                             engine='python')

    # 对于测试集，他的第一行是一个分隔，不读入
    test_data = pd.read_csv("data/adult_test.csv",
                            header=None,
                            names=headers,
                            na_values=["?"],
                            engine='python',
                            skiprows=1)

    all_data = train_data.append(test_data)
    print(all_data.head())

    print(all_data.describe())
