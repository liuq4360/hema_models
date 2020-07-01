from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import os
import random
import json


"""
    该脚本为每个用户推荐可行的车，作为客服或者销售人员对用户的精准推荐.
    
"""

# 给用户推荐的车的数量
K = 5

cwd = os.getcwd()  # 获取当前工作目录
data_path = os.path.abspath(os.path.join(cwd, "../../data"))  # 获取data目录

df_cluster = pd.read_csv(data_path + '/' + r'kmeans.csv')

# 每个id对应的cluster的映射字典。
id_cluster_dict = dict(df_cluster.values)

tmp = df_cluster.values
cluster_ids_dict = {}
for i in range(tmp.shape[0]):
    [id_, cluster_] = tmp[i]
    if cluster_ in cluster_ids_dict.keys():
        cluster_ids_dict[cluster_] = cluster_ids_dict[cluster_] + [id_]
    else:
        cluster_ids_dict[cluster_] = [id_]


# 一共有多少个类
# cluster_num = len(cluster_ids_dict)
# 打印出每一个类有多少个元素，即每类有多少辆车
for x, y in cluster_ids_dict.items():
    print("cluster " + str(x) + " : " + str(len(y)))

source_df = pd.read_csv(data_path + '/' + r'hema_car-new.csv')

car_info_df = source_df[['id', 'brand_name', 'series_name',
                         'model_name', 'color', 'sell_price', 'tag_names', 'displacement', ]]


# 获得每辆车的基础信息，这样可以直观看到推荐的结果，了解车的基本情况
car_info_dic = {}
for i in range(car_info_df.shape[0]):
    id_ = car_info_df['id'].values[i]
    brand_name = car_info_df['brand_name'].values[i]
    series_name = car_info_df['series_name'].values[i]
    model_name = car_info_df['model_name'].values[i]
    color = car_info_df['color'].values[i]
    sell_price = car_info_df['sell_price'].values[i]
    tag_names = car_info_df['tag_names'].values[i]
    displacement = car_info_df['displacement'].values[i]
    dic_ = {"brand_name": brand_name, "series_name": series_name, "model_name": model_name,
            "color": color, "sell_price": sell_price, "tag_names": tag_names, "displacement": displacement}
    car_info_dic[id_] = dic_


# 基于用户最近浏览的车，给他推荐相似的K个车。
def user_rec(car_id):
    return random.sample(cluster_ids_dict.get(id_cluster_dict.get(car_id)), K)


# 基于用户最近浏览的车，给他推荐相似的K个车及用户喜欢的车的信息和推荐的车的信息。
def user_rec_and_car_info(car_id):
    rec = random.sample(cluster_ids_dict.get(id_cluster_dict.get(car_id)), K)
    print("用户喜欢的车：" + str(car_id) + " -> " + str(car_info_dic[car_id]))
    print("给用户推荐的车：")
    for id_rec in rec:
        print("推荐的车：" + str(id_rec) + " -> " + str(car_info_dic[id_rec]))


user_rec_and_car_info(43)
