# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author : Kesheng Zhang
@B站：萧然哔哩哔
@Email : zks0053@163.com
@Time : 2022/3/13 19:23
@File : time_series_segmentation.py
@Software: PyCharm

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

def array_remove(myarray, index):
    ''' 将该数组中第 （index） 个索引对应的值，移出去'''
    list_rx = list(myarray)
    list_rx.remove(myarray[index])
    seg_rx = np.array(list_rx)
    return seg_rx


# num_segments = 3
# forceplot = 1
def bottomUp(data, num_segments, forceplot):
    '''
    :param data: 一维目标时间序列
    :param num_segments: 时间序列数据要划分的目标段数
    :param forceplot: 是否要绘制图形。forceplot=1，表示绘图
    :return: 返回数据分割的左右端点，其中左端点、右端点各在一个list中。
    '''

    left_x = np.arange(1, data.shape[0], 2)
    right_x = left_x + 1
    right_x[-1] = data.shape[0]  # 解决奇偶数组问题

    number_of_segments = len(left_x);

    seg_lx = left_x
    seg_rx = right_x
    seg_mc = np.ones(left_x.shape) * np.inf

    for i in range(number_of_segments - 1):
        lin_reg = LinearRegression()
        train_x = np.arange(seg_lx[i], seg_rx[i + 1], 1)
        train_x = train_x.reshape(-1, 1)
        train_y = data[seg_lx[i]:seg_rx[i + 1]]
        lin_reg.fit(train_x, train_y)
        y_hat = lin_reg.predict(train_x)
        seg_mc[i] = mean_squared_error(train_y, y_hat)

    while len(seg_mc) > num_segments:
        value = min(seg_mc)
        i = np.argmin(seg_mc)
        if i > 0 and i < (len(seg_mc) - 2):
            lin_reg = LinearRegression()
            train_x = np.arange(seg_lx[i], seg_rx[i + 2], 1)
            train_x = train_x.reshape(-1, 1)
            train_y = data[seg_lx[i]:seg_rx[i + 2]]
            lin_reg.fit(train_x, train_y)
            y_hat = lin_reg.predict(train_x)
            seg_mc[i] = mean_squared_error(train_y, y_hat)
            seg_rx[i] = seg_rx[i + 1]

            seg_lx = array_remove(seg_lx, i + 1)
            seg_rx = array_remove(seg_rx, i + 1)
            seg_mc = array_remove(seg_mc, i + 1)

            i = i - 1
            lin_reg = LinearRegression()
            train_x = np.arange(seg_lx[i], seg_rx[i + 1], 1)
            train_x = train_x.reshape(-1, 1)
            train_y = data[seg_lx[i]:seg_rx[i + 1]]
            lin_reg.fit(train_x, train_y)
            y_hat = lin_reg.predict(train_x)
            seg_mc[i] = mean_squared_error(train_y, y_hat)
        elif i == 0:
            lin_reg = LinearRegression()
            train_x = np.arange(seg_lx[i], seg_rx[i + 2], 1)
            train_x = train_x.reshape(-1, 1)
            train_y = data[seg_lx[i]:seg_rx[i + 2]]
            lin_reg.fit(train_x, train_y)
            y_hat = lin_reg.predict(train_x)
            seg_mc[i] = mean_squared_error(train_y, y_hat)
            seg_rx[i] = seg_rx[i + 1]
            seg_lx = array_remove(seg_lx, i + 1)
            seg_rx = array_remove(seg_rx, i + 1)
            seg_mc = array_remove(seg_mc, i + 1)
        else:
            seg_rx[i] = seg_rx[i + 1]
            seg_mc[i] = np.inf

            seg_lx = array_remove(seg_lx, i + 1)
            seg_rx = array_remove(seg_rx, i + 1)
            seg_mc = array_remove(seg_mc, i + 1)

            i = i - 1
            lin_reg = LinearRegression()
            train_x = np.arange(seg_lx[i], seg_rx[i + 1], 1)
            train_x = train_x.reshape(-1, 1)
            train_y = data[seg_lx[i]:seg_rx[i + 1]]
            lin_reg.fit(train_x, train_y)
            y_hat = lin_reg.predict(train_x)
            seg_mc[i] = mean_squared_error(train_y, y_hat)

    # print('seg_lx:', seg_lx)
    # print('seg_rx:', seg_rx)
    if forceplot == 1:
        plt.plot(data)
        plt.show()

    if forceplot == 1:
        plt.plot(data)

    for i in range(len(seg_mc)):

        train_x = np.arange(seg_lx[i], seg_rx[i], 1)
        train_x = train_x.reshape(-1, 1)
        train_y = data[seg_lx[i]:seg_rx[i]]
        lin_reg = LinearRegression()
        lin_reg.fit(train_x, train_y)
        y_hat = lin_reg.predict(train_x)

        if forceplot == 1:
            plt.plot(train_x, y_hat)
            plt.axvline(x=seg_rx[i], linestyle='--', color='orange')
            plt.axvline(x=seg_lx[i], linestyle='--', color='orange')
    plt.show()

    return seg_lx, seg_rx

