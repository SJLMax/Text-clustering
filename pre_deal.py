#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : pre_deal.py
# @Author: Shang
# @Date  : 2020/6/30

import os
import codecs
from itertools import groupby
import jieba
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import random
import xlsxwriter

def readtxt(path):
    with open(path,'r',encoding='utf8') as f:
        lines=f.readlines()
        lines=[i.replace('\xa0','').replace('\u3000','').replace('\n','') for i in lines]
        # print(lines)
        return lines


def deal(path):
    dir = os.listdir(path)
    print(dir)
    path_list = []
    for i in dir:
        path_list.append(path + str(i))
    print(path_list)
    output = codecs.open('./data.txt', 'w', encoding='utf8')
    for p, d in zip(path_list, dir):
        new_data = []
        data = readtxt(p)
        result = [list(g) for k, g in groupby(data, lambda x: x == '') if not k]
        # for k, g in groupby(data, lambda x: x == '\n'):
        #     print(k,list(g))
        # print(result)
        for line in result:
            t = ''.join(line)
            new_data.append(t)
        sentences=cutwords(new_data)
        for line in sentences:
            output.write(str(d[:2]) + '\t' + line + '\n')
    # print(new_data)


def cutwords(data):
    sentences=[]
    for line in data:
        slist = jieba.cut(line)
        output = " ".join(slist)
        # print(output.split(' '))
        f=''
        for word in output.split(' '):
            if word not in stoplist and word!='':
                f=f+' '+word
        sentences.append(f)
    return sentences

def kfold(corpus):
    random.shuffle(corpus)
    KF = KFold(n_splits=10, shuffle=False, random_state=100)
    model = RandomForestClassifier()
    i=0
    for train_index, test_index in KF.split(corpus):
        print("train_index:{},test_index:{}".format(train_index, test_index))
        data_train=corpus[list(train_index)[0]:list(train_index)[-1]]
        data_test=corpus[list(test_index)[0]:list(test_index)[-1]]
        p=r'./class/data_'+str(i)
        if not os.path.exists(p):
            os.makedirs(p)
        path='./class/data_'+str(i)+'/train.txt'
        path_test='./class/data_'+str(i)+'/test.txt'
        print(path)

        writetxt(path,data_train)
        writetxt(path_test,data_test)
        i+=1


def writetxt(path,data):
    with open(path,'w',encoding='utf8') as f:
        for i in data:
            f.write(str(i)+'\n')
    f.close()


if __name__ == '__main__':
    path='./人民日报分类1000条/'
    stoplist=readtxt('./哈工大停用词表.txt')

    # cutwords(corpus)
    # deal(path)
    corpus = readtxt('./data.txt')
    cluster=[]
    label=[]
    for l in corpus:
        l=l.replace('\n','')
        l=l.split('\t')
        cluster.append(l[1])
        if l[0]=='体育':
            label.append(0)
        elif l[0]=='副刊':
            label.append(1)
        elif l[0]=='国际':
            label.append(2)
        elif l[0]=='政治':
            label.append(3)
        elif l[0]=='文化':
            label.append(4)
        elif l[0]=='理论':
            label.append(5)
        elif l[0]=='社会':
            label.append(6)
        elif l[0]=='经济':
            label.append(7)
        elif l[0]=='要闻':
            label.append(8)
        elif l[0]=='评论':
            label.append(9)
    # writetxt(cluster,'./cluster.txt')
    # writetxt(label,'./label.txt')
    workbook = xlsxwriter.Workbook('./语料.xlsx')
    worksheet = workbook.add_worksheet('sheet1')
    worksheet.write(0, 0, '类别')
    worksheet.write(0, 1, 'cut_review')
    j=1
    for k, i in zip(cluster,label):
        worksheet.write(j, 0, i)
        worksheet.write(j, 1, k)
        j+=1
    workbook.close()







