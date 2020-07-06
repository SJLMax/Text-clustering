from sklearn.cluster import KMeans
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from itertools import cycle  ##python自带的迭代器模块
from sklearn.metrics import silhouette_score
from sklearn import metrics
import pca
import pre_deal

X=pca.pca()
lables_true=pre_deal.readtxt('./label.txt')

# kmeans聚类
def kmeans():
    kmeans = KMeans(n_clusters=10).fit(X)
    lables = kmeans.labels_
    s = silhouette_score(X,lables, metric='euclidean')
    print('silhouette_score:',s)
    # precision
    sum=0
    for x,y in zip(lables_true,lables):
        print(x,y)
        if str(x)==str(y):
            sum+=1
    print('p:',sum/len(lables))
    plt.figure(1)
    plt.clf()
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(10), colors):
        # 根据lables中的值是否等于k，重新组成一个True、False的数组
        my_members = lables == k
        # X[my_members, 0] 取出my_members对应位置为True的值的横坐标
        plt.plot(X[my_members, 0], X[my_members, 1], col + '.')

    plt.title('Estimated number of clusters: %d' % 10)
    plt.show()


# 层次聚类
def hc():
    # 产生随机数据的中心
    # centers = [[1, 1], [-1, 2], [1, 1]]
    # 产生的数据个数
    # n_samples=3000
    # 生产数据
    # X, lables_true = make_blobs(n_samples=n_samples, centers= centers, cluster_std=0.6, random_state =0)
    # 设置分层聚类函数
    linkages = ['ward', 'average', 'complete']
    n_clusters_ = 10
    ac = AgglomerativeClustering(linkage=linkages[2],n_clusters = n_clusters_)
    ##训练数据
    ac.fit(X)
    ##每个数据的分类
    lables = ac.labels_
    # print(lables_true)
    print(lables)

    s = silhouette_score(X, ac.labels_, metric='euclidean')
    print('silhouette_score:',s)
    # precision
    sum=0
    for x,y in zip(lables_true,lables):
        # print(x,y)
        if str(x)==str(y):
            sum+=1
    print('p:',sum/len(lables))

    # 绘图
    plt.figure(1)
    plt.clf()
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        ##根据lables中的值是否等于k，重新组成一个True、False的数组
        my_members = lables == k
        ##X[my_members, 0] 取出my_members对应位置为True的值的横坐标
        plt.plot(X[my_members, 0], X[my_members, 1], col + '.')

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


if __name__ == '__main__':
    # hc()
    kmeans()




