# 这里实现一个简单的协同过滤推荐算法
import pandas as pd
import numpy as np
from pprint import pprint
# 直接计算某两项的杰卡德相似系数   两个集合的交集元素个数在并集中所占的比例, 非常适用于布尔向量表示
from sklearn.metrics import jaccard_score

# 计算所有的数据两两的杰卡德相似系数


from sklearn.metrics.pairwise import pairwise_distances
def data_process():
    # 用户  横坐标对应的
    users = ["User1", "User2", "User3", "User4", "User5"]
    # 商品  纵坐标对应的
    items = ["Item A", "Item B", "Item C", "Item D", "Item E"]
    # 构建数据集
    # datasets = [
    #     ["buy",None,"buy","buy",None],
    #     ["buy",None,None,"buy","buy"],
    #     ["buy",None,"buy",None,None],
    #     [None,"buy",None,"buy","buy"],
    #     ["buy","buy","buy",None,"buy"],
    # ]
    # 肯定要让数据方便处理
    # 用户购买记录数据集
    datasets = [
        [1,0,1,1,0],
        [1,0,0,1,1],
        [1,0,1,0,0],
        [0,1,0,1,1],
        [1,1,1,0,1],
    ]
    # 构造矩阵  横坐标代表商品，纵坐标代表用户
    df = pd.DataFrame(datasets,
                  columns=items,
                  index=users)
    print(df)
    # 计算Item A与Item B之间的相似度  “跟你喜欢的东西**相似的东西**你也很有可能喜欢 ”：基于物品的协同过滤推荐（Item-based CF）
    # print(f'Item A与Item B之间的相似度为: {jaccard_score(df["Item A"], df["Item B"])}')
    # print('=============================')

    
    # 计算用户间相似度

    # user_similar = 1 - pairwise_distances(df, metric="jaccard")#报错 
    # “跟你喜好**相似的人**喜欢的东西你也很有可能喜欢” ：基于用户的协同过滤推荐（User-based CF）
    user_similar = 1 - pairwise_distances(df.values, metric='jaccard')
    user_similar = pd.DataFrame(user_similar, columns=users, index=users)
    
    print("用户之间的两两相似度：")
    # 适用于基于用户的协同过滤推荐（User-based CF）。
    print(user_similar)
# 用户之间的两两相似度：这个矩阵相当于中心对称的
#           User1  User2     User3  User4  User5
# User1  1.000000   0.50  0.666667    0.2    0.4  
# User2  0.500000   1.00  0.250000    0.5    0.4
# User3  0.666667   0.25  1.000000    0.0    0.5
# User4  0.200000   0.50  0.000000    1.0    0.4
# User5  0.400000   0.40  0.500000    0.4    1.0

    
    # 计算物品间相似度  df.T 返回 df 的转置版本，其中行变为列，列变为行。
    # 使用 pairwise_distances(df.T.values, metric='jaccard') 计算的是商品之间的杰卡德相似度。具体来说，它计算的是每两个商品之间的购买记录的相似度。
    # “跟你喜欢的东西**相似的东西**你也很有可能喜欢 ”：基于物品的协同过滤推荐（Item-based CF）
    item_similar = 1 - pairwise_distances(df.T.values, metric="jaccard")
    item_similar = pd.DataFrame(item_similar, columns=items, index=items)
    print("物品之间的两两相似度：")
    print(item_similar)
    # 适用于基于物品的协同过滤推荐（Item-based CF）。
    return df, user_similar, item_similar

# my_dict = {
#     "apple": 1,
#     "banana": 2,
#     "orange": 3
# }

# # 打印 dict 内容
# for key, value in my_dict.items():
#     print(f"{key}: {value}")

# “跟你喜好**相似的人**喜欢的东西你也很有可能喜欢” ：基于用户的协同过滤推荐（User-based CF）
def user_similar_process(df, user_similar):
    topN_users = dict()#dict()
    # 遍历每一行数据
    for i in user_similar.index:
        # print(f'{i}\t', end='')#User1   User2   User3   User4   User5
        # 取出每一列数据，并删除自身，然后排序数据
        _df = user_similar.loc[i].drop([i])
        # print(_df.values)
        _df_sorted = _df.sort_values(ascending=False)
        # print(_df_sorted.values)

        top2 = list(_df_sorted.index[:2])#取前两个  比如与User1有相似爱好的用户为User3与User2
        # print(top2)
        
        topN_users[i] = top2
        # print(topN_users[i])

    print("Top2相似用户：")
    pprint(topN_users)

    rs_results = dict()
    # # 构建推荐结果
    for user, sim_users in topN_users.items():
        rs_result = set()    # 存储推荐结果
        for sim_user in sim_users:#遍历比如['User3', 'User2']
            # 构建初始的推荐结果
            # loc 是 DataFrame 的一个索引器，用于通过标签访问行或列。
            # sim_user 是一个相似用户的名称，通过 df.loc[sim_user] 可以获取该用户的购买记录。
            # print(f'df.loc[{sim_user}]的值为:{df.loc[sim_user]}')#df.loc[User3]的值为:Item A 1 
            # replace(0, np.nan) 将用户购买记录中所有的 0（表示未购买）替换为 NaN（表示缺失值）。
            # dropna() 方法删除所有包含 NaN 的条目。
            rs_result = rs_result.union(set(df.loc[sim_user].replace(0,np.nan).dropna().index))
            # union 是集合的一个方法，用于返回两个集合的并集（即包含所有元素的集合，不包含重复元素）。
            # rs_result.union(...) 将当前的推荐结果集合 rs_result 和相似用户购买的物品集合合并，生成一个新的集合。
            print(f'rs_result的结果为{rs_result}')
        # 过滤掉已经购买过的物品
        rs_result -= set(df.loc[user].replace(0,np.nan).dropna().index)
        rs_results[user] = rs_result
    print("User-based最终推荐结果: ")
    pprint(rs_results)



# “跟你喜欢的东西**相似的东西**你也很有可能喜欢 ”：基于物品的协同过滤推荐（Item-based CF）
def item_similar_process(df, item_similar):
    topN_items = {}
    # 遍历每一行数据
    for i in item_similar.index:
        # 取出每一列数据，并删除自身，然后排序数据
        _df = item_similar.loc[i].drop([i])
        _df_sorted = _df.sort_values(ascending=False)

        top2 = list(_df_sorted.index[:2])
        topN_items[i] = top2

    print("Top2相似物品：")
    pprint(topN_items)

    rs_results = {}
    # 构建推荐结果
    for user in df.index:    # 遍历所有用户
        rs_result = set()
        for item in df.loc[user].replace(0,np.nan).dropna().index:   # 取出每个用户当前已购物品列表
            # 根据每个物品找出最相似的TOP-N物品，构建初始推荐结果
            rs_result = rs_result.union(topN_items[item])
        # 过滤掉用户已购的物品
        rs_result -= set(df.loc[user].replace(0,np.nan).dropna().index)
        # 添加到结果中
        rs_results[user] = rs_result

    print("Item-based最终推荐结果：")
    pprint(rs_results)

# 这里我们选择使用杰卡德相似系数[0,1]计算相似度
# 在前面的demo中，我们只是使用用户对物品的一个购买记录，类似也可以是比如浏览点击记录、收听记录等等。这样数据我们预测的结果其实相当于是在预测用户是否对某物品感兴趣，对于喜好程度不能很好的预测。

# 因此在协同过滤推荐算法中其实会更多的利用用户对物品的“评分”数据来进行预测，通过评分数据集，我们可以预测用户对于他没有评分过的物品的评分。其实现原理和思想和都是一样的，只是使用的数据集是用户-物品的评分数据。
def main():
    # print('===')
    df, user_similar, item_similar = data_process()#获得用户之间两两的相似度  有了两两的相似度，接下来就可以筛选TOP-N相似结果，并进行推荐了
    user_similar_process(df, user_similar)#User-Based CF
    item_similar_process(df, item_similar)#Item-Based CF
# 22440000
# 13203359226
if __name__ == "__main__":
    main()