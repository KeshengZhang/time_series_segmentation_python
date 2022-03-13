# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author : Kesheng Zhang
@B站：萧然哔哩哔
@Email : zks0053@163.com
@Time : 2022/3/13 19:27
@File : main.py
@Software: PyCharm

"""

import numpy as np
from time_series_segmentation import bottomUp

data_input = np.load('testData1_timeseriesSegmentation.npy')
# data_input = np.load('testData2_timeseriesSegmentation.npy')
# data_input = np.load('testData3_timeseriesSegmentation.npy')
# data_input = np.load('testData4_timeseriesSegmentation.npy')


'''
bottomUp()函数介绍
:param data: 一维目标时间序列
:param num_segments: 时间序列数据要划分的目标段数
:param forceplot: 是否要绘制图形。forceplot=1，表示绘图
:return: 返回数据分割的左右端点，其中左端点、右端点各在一个list中。
'''
seg_lx, seg_rx = bottomUp(data=data_input, num_segments=6, forceplot=1)

print('seg_lx:', seg_lx)
print('seg_rx:', seg_rx)
