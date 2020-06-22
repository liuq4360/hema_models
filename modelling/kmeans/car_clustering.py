from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import os

"""
    本脚本是利用scikit-learn中的KMeans函数进行聚类。
    最终的结果有两列，一列id，另外一列是该id对应的聚类。
    聚类结果存在data目录下的kmeans.csv中。
    
"""

# 聚类的数量
n_clusters = 1000

cwd = os.getcwd()  # 获取当前工作目录
data_path = os.path.abspath(os.path.join(cwd, "../../data"))  # 获取data目录

df_train = pd.read_csv(data_path + '/' + r'train.csv')


# X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
# k_means = KMeans(n_clusters=2, random_state=0).fit(X)

# n_clusters: 一共聚多少类，默认值8
# init：选择中心点的初始化方法，默认值k-means++
# n_init：算法基于不同的中心点运行多少次，最后的结果基于最好的一次迭代的结果，默认值10
# max_iter: 最大迭代次数，默认值300
k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10,
                 max_iter=300).fit(df_train.drop(columns=['id']).values)

# 训练样本中每条记录所属的类别
print(k_means.labels_)
# 预测某个样本属于哪个聚类
# print(k_means.predict(np.random.rand(1, df_train.shape[1])))
print(k_means.predict(np.random.randint(20, size=(2, df_train.drop(columns=['id']).shape[1]))))
# 每个聚类的聚类中心
print(k_means.cluster_centers_)

result_array = np.hstack((np.asarray([df_train['id'].values]).T,
                          np.asarray([k_means.labels_]).T))

# 将车辆id和具体的类别转化为DataFrame。
cluster_result = DataFrame(result_array, columns=['id', 'cluster'])

# index = 0 写入时不保留索引列。
cluster_result.to_csv(data_path + '/' + r'kmeans.csv', index=0)
# read
# cluster_result = pd.read_csv(data_path + '/' + r'kmeans.csv')
