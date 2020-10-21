#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : pca.py
# @Author: Shang
# @Date  : 2020/7/1

import numpy as np
from sklearn.decomposition import PCA


def pca():
    with open('./w2v_100D_ns.txt','r',encoding='utf8') as f:
        line = f.readline()
        data_list = []
        while line:
            num = list(map(float,line.split()))
            data_list.append(num)
            line = f.readline()
        f.close()
    X = np.array(data_list)  # 导入数据
    pca = PCA(n_components=2)   # 降到2维
    pca.fit(X)                  # 训练
    newX=pca.fit_transform(X)   # 降维后的数据
    # PCA(copy=True, n_components=2, whiten=False)
    # print(pca.explained_variance_ratio_)  #输出贡献率
    print(newX)
    return newX
