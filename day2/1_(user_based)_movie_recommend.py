# 刚刚开始基本上都是冷启动，只有到后期才能进行协同过滤

# 首先实现基于用户的协同过滤

import os

import pandas as pd
import numpy as np
import pprint

# 加载数据集
DATA_PATH = "./ml-latest-small/ratings.csv"

# 数据集缓存地址
CACHE_DIR = "./cache/"

def load_data(data_path):

    '''
    加载数据
    :param data_path: 数据集路径
    :param cache_path: 数据集缓存路径
    :return: 用户-物品评分矩阵
    '''
    # 检查CACHE_DIR文件夹是否存在
    if not os.path.exists(CACHE_DIR):
        # 如果文件夹不存在，则创建文件夹
        os.makedirs(CACHE_DIR)
        print(f"Folder '{CACHE_DIR}' created.")
    else:
        # 如果文件夹已经存在，则什么都不做
        print(f"Folder '{CACHE_DIR}' already exists.")

    # 数据集缓存地址
    cache_path = os.path.join(CACHE_DIR, "ratings_matrix.cache")

    print("开始加载数据集...")
    if os.path.exists(cache_path):    # 判断是否存在缓存文件
        print("加载缓存中...")
        ratings_matrix = pd.read_pickle(cache_path)
        print("从缓存加载数据集完毕")
    else:
        print("加载新数据中...")
        # 设置要加载的数据字段的类型
        dtype = {"userId": np.int32, "movieId": np.int32, "rating": np.float32}
        # 加载数据，我们只用前三列数据，分别是用户ID，电影ID，已经用户对电影的对应评分
        ratings = pd.read_csv(data_path, dtype=dtype, usecols=range(3))
        # 透视表，将电影ID转换为列名称，转换成为一个User-Movie的评分矩阵
        ratings_matrix = ratings.pivot_table(index=["userId"], columns=["movieId"], values="rating")
        # 存入缓存文件
        ratings_matrix.to_pickle(cache_path)
        print("数据集加载完毕")
    return  ratings_matrix

# 计算用户或物品两两相似度：  默认计算用户的相似度
def compute_pearson_similarity(ratings_matrix, based="user"):
    '''
    计算皮尔逊相关系数
    :param ratings_matrix: 用户-物品评分矩阵
    :param based: "user" or "item"
    :return: 相似度矩阵
    '''
    user_similarity_cache_path = os.path.join(CACHE_DIR, "user_similarity.cache")
    item_similarity_cache_path = os.path.join(CACHE_DIR, "item_similarity.cache")
    # 基于皮尔逊相关系数计算相似度
    # 用户相似度
    if based == "user":
        if os.path.exists(user_similarity_cache_path):
            print("正从缓存加载用户相似度矩阵")
            similarity = pd.read_pickle(user_similarity_cache_path)
        else:
            print("开始计算用户相似度矩阵")
            similarity = ratings_matrix.T.corr()
            similarity.to_pickle(user_similarity_cache_path)

    elif based == "item":
        if os.path.exists(item_similarity_cache_path):
            print("正从缓存加载物品相似度矩阵")
            similarity = pd.read_pickle(item_similarity_cache_path)
        else:
            print("开始计算物品相似度矩阵")
            similarity = ratings_matrix.corr()
            similarity.to_pickle(item_similarity_cache_path)
    else:
        raise Exception("Unhandled 'based' Value: %s"%based)
    print("相似度矩阵计算/加载完毕")
    return similarity

def predict(uid, iid, ratings_matrix, user_similar):
    '''
    预测给定用户对给定物品的评分值  这个是有一个评分值的运算公式
    :param uid: 用户ID
    :param iid: 物品ID
    :param ratings_matrix: 用户-物品评分矩阵
    :param user_similar: 用户两两相似度矩阵
    :return: 预测的评分值
    '''
    print("开始预测用户<%d>对电影<%d>的评分..."%(uid, iid))
    # 1. 找出uid用户的相似用户 把NaN去掉
    similar_users = user_similar[uid].drop([uid]).dropna()
    # 相似用户筛选规则：正相关的用户
    similar_users = similar_users.where(similar_users>0).dropna()
    if similar_users.empty is True:
        raise Exception("用户<%d>没有相似的用户" % uid)

    # 2. 从uid用户的近邻相似用户中筛选出对iid物品有评分记录的近邻用户
    ids = set(ratings_matrix[iid].dropna().index)&set(similar_users.index)
    finally_similar_users = similar_users.loc[list(ids)]

    # 3. 结合uid用户与其近邻用户的相似度预测uid用户对iid物品的评分
    sum_up = 0    # 评分预测公式的分子部分的值
    sum_down = 0    # 评分预测公式的分母部分的值
    for sim_uid, similarity in finally_similar_users.items():
        # 近邻用户的评分数据
        sim_user_rated_movies = ratings_matrix.loc[sim_uid].dropna()
        # 近邻用户对iid物品的评分
        sim_user_rating_for_item = sim_user_rated_movies[iid]
        # 计算分子的值
        sum_up += similarity * sim_user_rating_for_item
        # 计算分母的值
        sum_down += similarity

    # 计算预测的评分值并返回
    predict_rating = sum_up/sum_down
    print("预测出用户<%d>对电影<%d>的评分：%0.2f" % (uid, iid, predict_rating))
    return round(predict_rating, 2)


# 预测每个人对每部电影的评分   非常消耗时间  没必要
def predict_all(uid, ratings_matrix, user_similar):
    '''
    预测全部评分
    :param uid: 用户id
    :param ratings_matrix: 用户-物品打分矩阵
    :param user_similar: 用户两两间的相似度
    '''
    # 准备要预测的物品的id列表
    item_ids = ratings_matrix.columns
    # 逐个预测
    for iid in item_ids:
        try:
            rating = predict(uid, iid, ratings_matrix, user_similar)
        except Exception as e:
            print(e)
        else:
            yield uid, iid, rating


# yield 是一个生成器语法，用于生成一个生成器对象。与 return 不同的是，yield 会在函数被调用时返回一个值，并且在后续调用时能够继续执行，从而可以生成一系列值。

# 在这个例子中，yield 会一次返回 (uid, iid, rating) 元组，这样你可以逐个获取所有预测的评分，而不需要一次性计算并返回所有结果。这对于处理大数据集非常有用，因为它允许你在需要时逐步计算和获取结果。
def _predict_all(uid, item_ids, ratings_matrix, user_similar):
    '''
    预测全部评分
    :param uid: 用户id
    :param item_ids: 要预测的物品id列表
    :param ratings_matrix: 用户-物品打分矩阵
    :param user_similar: 用户两两间的相似度
    :return: 生成器，逐个返回预测评分
    '''
    # 逐个预测
    for iid in item_ids:
        try:
            rating = predict(uid, iid, ratings_matrix, user_similar)
        except Exception as e:
            print(e)
        else:
            yield uid, iid, rating

def predict_all(uid, ratings_matrix, user_similar, filter_rule=None):
    '''
    预测全部评分，并可根据条件进行前置过滤
    :param uid: 用户ID
    :param ratings_matrix: 用户-物品打分矩阵
    :param user_similar: 用户两两间的相似度
    :param filter_rule: 过滤规则，只能是四选一，否则将抛异常："unhot","rated",["unhot","rated"],None
    :return: 生成器，逐个返回预测评分
    可以是 "unhot"（过滤非热门电影）、"rated"（过滤用户评分过的电影）、["unhot","rated"]（同时过滤非热门和用户评分过的电影）、或者 None（不过滤）。
    '''
    # 如果没有提供过滤规则，直接获取所有物品的 ID 列表（即矩阵的列）。
    if not filter_rule:
        item_ids = ratings_matrix.columns
    # isinstance(filter_rule, str) 函数用于检查变量 filter_rule 是否是字符串类型。
    elif isinstance(filter_rule, str) and filter_rule == "unhot":
        '''过滤非热门电影'''
        # 统计每部电影的评分数
        count = ratings_matrix.count()
        # 过滤出用户对某个电影的评分数量高于10的电影，作为热门电影
        item_ids = count.where(count>10).dropna().index
    elif isinstance(filter_rule, str) and filter_rule == "rated":
        '''过滤用户评分过的电影'''
        # 获取用户对所有电影的评分记录
        user_ratings = ratings_matrix.loc[uid]
        # 评分范围是1-5，小于6的都是评分过的，除此以外的都是没有评分的
        _ = user_ratings<6
        item_ids = _.where(_==False).dropna().index
    elif isinstance(filter_rule, list) and set(filter_rule) == set(["unhot", "rated"]):
        '''过滤非热门和用户已经评分过的电影'''
        count = ratings_matrix.count()
        ids1 = count.where(count > 10).dropna().index

        user_ratings = ratings_matrix.loc[uid]
        _ = user_ratings < 6
        ids2 = _.where(_ == False).dropna().index
        # 取二者交集
        item_ids = set(ids1)&set(ids2)
    else:
        raise Exception("无效的过滤参数")

    yield from _predict_all(uid, item_ids, ratings_matrix, user_similar)

# 根据预测评分为指定用户进行TOP-N推荐：
def top_k_rs_result(k):
    ratings_matrix = load_data(DATA_PATH)
    user_similar = compute_pearson_similarity(ratings_matrix, based="user")
    results = predict_all(1, ratings_matrix, user_similar, filter_rule=["unhot", "rated"])
    return sorted(results, key=lambda x: x[2], reverse=True)[:k]


if __name__ == '__main__':
    ratings_matrix = load_data(DATA_PATH)
# 一共610个用户  每个用户会对至少20部电影进行评价，但是肯定有很多电影某个用户是没有看过的  评分就是NaN  电影编号最大为193609
# movieId  1       2       3       4       5       6       ...  193579  193581  193583  193585  193587  193609
# userId                                                   ...                                                
# 1           4.0     NaN     4.0     NaN     NaN     4.0  ...     NaN     NaN     NaN     NaN     NaN     NaN
# 2           NaN     NaN     NaN     NaN     NaN     NaN  ...     NaN     NaN     NaN     NaN     NaN     NaN
# 3           NaN     NaN     NaN     NaN     NaN     NaN  ...     NaN     NaN     NaN     NaN     NaN     NaN
# 4           NaN     NaN     NaN     NaN     NaN     NaN  ...     NaN     NaN     NaN     NaN     NaN     NaN
# 5           4.0     NaN     NaN     NaN     NaN     NaN  ...     NaN     NaN     NaN     NaN     NaN     NaN
# ...         ...     ...     ...     ...     ...     ...  ...     ...     ...     ...     ...     ...     ...
# 606         2.5     NaN     NaN     NaN     NaN     NaN  ...     NaN     NaN     NaN     NaN     NaN     NaN
# 607         4.0     NaN     NaN     NaN     NaN     NaN  ...     NaN     NaN     NaN     NaN     NaN     NaN
# 608         2.5     2.0     2.0     NaN     NaN     NaN  ...     NaN     NaN     NaN     NaN     NaN     NaN
# 609         3.0     NaN     NaN     NaN     NaN     NaN  ...     NaN     NaN     NaN     NaN     NaN     NaN
# 610         5.0     NaN     NaN     NaN     NaN     5.0  ...     NaN     NaN     NaN     NaN     NaN     NaN

# [610 rows x 9724 columns]
    # print(ratings_matrix)
    user_similar = compute_pearson_similarity(ratings_matrix, based="user")
    # 预测用户1对物品1的评分
    # predict(1, 1, ratings_matrix, user_similar)
    # 预测用户1对物品2的评分
    # predict(1, 2, ratings_matrix, user_similar)
    # print(user_similar)
    # item_similar = compute_pearson_similarity(ratings_matrix, based="item")
    # print(item_similar)
    # for i in predict_all(1, ratings_matrix, user_similar):
    #     pass
    # for result in predict_all(1, ratings_matrix, user_similar, filter_rule=["unhot", "rated"]):
    #     print(result)
    result = top_k_rs_result(20)
    pprint(result)