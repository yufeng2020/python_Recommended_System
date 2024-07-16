import os

import pandas as pd
import numpy as np
import pprint
import pandas as pd
import numpy as np


class BaselineCFBySGD(object):

    def __init__(self, number_epochs, alpha, reg, columns=["uid", "iid", "rating"]):
        # 梯度下降最高迭代次数  20
        self.number_epochs = number_epochs
        # 学习率  0.1
        self.alpha = alpha
        # 正则参数  0.1
        self.reg = reg
        # 数据集中user-item-rating字段的名称
        self.columns = columns

    def fit(self, dataset):
        '''
        :param dataset: uid, iid, rating
        :return:
        '''
        self.dataset = dataset
        # 用户评分数据  groupby以用户ID做分组  agg就是要将拿出来的用户ID放入到list  每一个用户对应很多电影编号  和  很多电影的评分
        self.users_ratings = dataset.groupby(self.columns[0]).agg([list])[[self.columns[1], self.columns[2]]]
        # 物品评分数据
        self.items_ratings = dataset.groupby(self.columns[1]).agg([list])[[self.columns[0], self.columns[2]]]
        # 计算全局平均分
        self.global_mean = self.dataset[self.columns[2]].mean()
        # 调用sgd方法训练模型参数
        self.bu, self.bi = self.sgd()

    def sgd(self):
        '''
        利用随机梯度下降，优化bu，bi的值
        :return: bu, bi
        '''
        # 初始化bu、bi的值，全部设为0  users_ratings.index就是self.columns[0]    self.items_ratings.index就是self.columns[1]
        # np.zeros生成全0的
        bu = dict(zip(self.users_ratings.index, np.zeros(len(self.users_ratings))))
        bi = dict(zip(self.items_ratings.index, np.zeros(len(self.items_ratings))))
        # itertuples 方法将数据集中的每一行转换为一个元组，index=False 表示不包括行索引。
        for i in range(self.number_epochs):
            print("iter%d" % i)
            for uid, iid, real_rating in self.dataset.itertuples(index=False):
                # real_rating 是实际评分。
                # self.global_mean 是全局平均评分。 bu[uid] 是用户 uid 的偏置项。  bi[iid] 是物品 iid 的偏置项。
                error = real_rating - (self.global_mean + bu[uid] + bi[iid])
                # self.alpha 是学习率，控制每次更新的步长。  self.reg 是正则化参数，用于防止过拟合。
                bu[uid] += self.alpha * (error - self.reg * bu[uid])
                bi[iid] += self.alpha * (error - self.reg * bi[iid])

        return bu, bi

    def predict(self, uid, iid):
        predict_rating = self.global_mean + self.bu[uid] + self.bi[iid]
        return predict_rating


if __name__ == '__main__':

    dtype = [("userId", np.int32), ("movieId", np.int32), ("rating", np.float32)]

    dataset = pd.read_csv("./ml-latest-small/ratings.csv", usecols=range(3), dtype=dict(dtype))

    bcf = BaselineCFBySGD(20, 0.1, 0.1, ["userId", "movieId", "rating"])
    bcf.fit(dataset)

    while True:
        uid = int(input("uid: "))
        iid = int(input("iid: "))
        print(bcf.predict(uid, iid))