import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.metrics.pairwise import pairwise_distances
# 在前面的demo中，我们只是使用用户对物品的一个购买记录，类似也可以是比如浏览点击记录、收听记录等等。这样数据我们预测的结果其实相当于是在预测用户是否对某物品感兴趣，对于喜好程度不能很好的预测。
# 因此在协同过滤推荐算法中其实会更多的利用用户对物品的“评分”数据来进行预测，通过评分数据集，我们可以预测用户对于他没有评分过的物品的评分。其实现原理和思想和都是一样的，只是使用的数据集是用户-物品的评分数据。

# 当评分是稠密的话，直接使用皮尔逊相关系数去进行计算
# dense_data_process dense_user_similar_process dense_item_similar_process
def dense_data_process():
    # 预测用户1对物品E的评分
    users = ["User1", "User2", "User3", "User4", "User5"]
    items = ["Item A", "Item B", "Item C", "Item D", "Item E"]
    # 用户购买记录数据集
    # 注意这里构建评分数据时，对于缺失的部分我们需要保留为None，如果设置为0那么会被当作评分值为0去对待
    datasets = [
        [5,3,4,4,None],
        [3,1,2,3,3],
        [4,3,4,3,5],
        [3,3,1,5,4],
        [1,5,5,2,1],
    ]
    # print()
    df = pd.DataFrame(datasets, columns=items, index=users)
    
    # print(df)
#        Item A  ...  Item E
# User1       5  ...     NaN
# User2       3  ...     3.0
# User3       4  ...     5.0
# User4       3  ...     4.0
# User5       1  ...     1.0
    # 直接计算皮尔逊相关系数
    # 默认是按列进行计算，因此如果计算用户间的相似度，当前需要进行转置
    print("用户之间的两两相似度：")
    user_similar = df.T.corr()
    print(user_similar.round(4))
#         User1   User2   User3   User4   User5
# User1  1.0000  0.8528  0.7071  0.0000 -0.7921
# User2  0.8528  1.0000  0.4677  0.4900 -0.9001
# User3  0.7071  0.4677  1.0000 -0.1612 -0.4666
# User4  0.0000  0.4900 -0.1612  1.0000 -0.6415
# User5 -0.7921 -0.9001 -0.4666 -0.6415  1.0000
# 可以发现与用户1最相似的是用户2与用户3  最不相似的是用户5
    
    
    # 计算物品之间两两的相似度
    print("物品之间的两两相似度：")
    item_similar = df.corr()
    print(item_similar.round(4))
# 
#         Item A  Item B  Item C  Item D  Item E
# Item A  1.0000 -0.4767 -0.1231  0.5322  0.9695
# Item B -0.4767  1.0000  0.6455 -0.3101 -0.4781
# Item C -0.1231  0.6455  1.0000 -0.7206 -0.4276
# Item D  0.5322 -0.3101 -0.7206  1.0000  0.5817
# Item E  0.9695 -0.4781 -0.4276  0.5817  1.0000
# 与物品A最相似的物品分别是物品E和物品D。 最不相干的物品B

# 我们在预测评分时，往往是通过与其有正相关的用户或物品进行预测，如果不存在正相关的情况，那么将无法做出预测。这一点尤其是在稀疏评分矩阵中尤为常见，因为稀疏评分矩阵中很难得出正相关系数。
    return df, user_similar, item_similar

#**User-Based CF 评分预测 要预测用户1对物品E的评分，那么可以根据与用户1最近邻的用户2和用户3进行预测
def dense_user_similar_process(df, user_similar):
    # 该评分的值就是0.85*3 + 0.71 * 5 / (0.85 + 0.71) = 3.91 
    # dense_data_process
    print()

# 要预测用户1对物品E的评分，那么可以根据与物品E最近邻的物品A和物品D进行预测
def dense_item_similar_process(df, item_similar):
    print()
    # 该评分的值就是 0.97 * 5 + 0.58 * 4 / (0.97 +0.58) = 4.63



# 当评分是稀疏的话，

def main():
    # print('===')
    # 当评分是稠密的话，直接使用皮尔逊相关系数去进行计算
    df, user_similar, item_similar = dense_data_process()#获得用户之间两两的相似度  有了两两的相似度，接下来就可以筛选TOP-N相似结果，并进行推荐了
    dense_user_similar_process(df, user_similar)#User-Based CF
    dense_item_similar_process(df, item_similar)#Item-Based CF
# User-Based CF预测评分和Item-Based CF的评分结果也是存在差异的，
# 因为严格意义上他们其实应当属于两种不同的推荐算法，各自在不同的领域不同场景下，
# 都会比另一种的效果更佳，但具体哪一种更佳，必须经过合理的效果评估，因此在实现推荐系统时这两种算法往往都是需要去实现的，然后对产生的推荐效果进行评估分析选出更优方案。
    
    # dense_data_process()
    

    # 当评分是稀疏的话

if __name__ == "__main__":
    main()




