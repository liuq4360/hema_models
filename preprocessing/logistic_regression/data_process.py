import pandas as pd
import csv
import numpy as np
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder
import math
from pandas import Series, DataFrame
import os

from sklearn.model_selection import train_test_split


cwd = os.getcwd()  # 获取当前工作目录
data_path = os.path.abspath(os.path.join(cwd, "../../data"))  # 获取data目录

"""
下面读取4类用于构建特征的原始数据
"""

# 二手车数据&用户相关数据
df_car = pd.read_csv(data_path + '/' + r'hema_car-new.csv')

# 浏览二手车源表，用户行为记录
df_event = pd.read_csv(data_path + '/' + r'csb_car_event.csv')

# 线上求购表。求购代表了用户的兴趣。
df_purchase = pd.read_csv(data_path + '/' + r'hema_car_purchase_info.csv')

# 用户登录记录表。
df_login = pd.read_csv(data_path + '/' + r'hema_user_login_record.csv')

# 针对df_car数据，车相关的特征就采用kmeans构建的特征train.csv，用户特征就用两个特征
# 一个是 user_province， 一个是 user_status，其他没有很合适的数据用于构建用户特征。


"""
下面开始构建各类特征
"""

# car 相关特征直接从kmeans中构建的取用，不另行构建特征了
# 将train.csv中的id重新命名为car_id方便后面对各类特征进行拼接join
data = df_car[['user_id', 'id',
               'brand_id', 'color', 'displacement', 'tag_ids',
               'mileage', 'sell_price', 'transfer_num', 'browse_num',
               'license_time',
               'user_province', 'user_status',
               'series_id', 'model_id', ]].rename(columns={"id": "car_id"})

# dropna(subset=['user_province'], inplace=True) 去掉user_province这一列的包含空值的行
# 用户特征从df_car中获取，一共31个省份，一共3种user_status，都可以采用one-hot编码。
# user_province中包含443条NaN，可以直接去掉。
data.dropna(subset=['user_province'], inplace=True)

# 特征工程

# license_time: 将上牌时间到现在过了多少年作为一个特征
data['past_year'] = data['license_time'].apply(lambda t: 2020 - int(str(t)[0:4]))

# mileage：最大值为172416.0，最小值为0，最大值取对数后为12. x -> log(x+1)
data['log_mileage'] = data['mileage'].apply(lambda t: math.log(t + 1))

# sell_price: 去掉价格高于400万的50条，可能是脏数据
data = data[data.sell_price < 400]
# sell_price 用min-max归一化
max_sell_price = data.sell_price.nlargest(1, keep='all').values[0]
min_sell_price = data.sell_price.nsmallest(1, keep='all').values[0]
data['min_max_sell_price'] = data['sell_price'].apply(
    lambda t: (t - min_sell_price) / (max_sell_price - min_sell_price))

# transfer_num: 最大值22，最小值0，用正态分布归一化. 17144个空值。 用0填充。
data['transfer_num'] = data['transfer_num'].apply(lambda t: 0 if np.isnan(t) else t)
mean_transfer_num = np.mean(data.transfer_num)
std_transfer_num = np.std(data.transfer_num)
data['gauss_transfer_num'] = data['transfer_num'].apply(lambda t: (t - mean_transfer_num) / std_transfer_num)

# browse_num, 数值特征，最大值1060，最小值0，行归一化。
sqrt_square_sum = math.sqrt(data.browse_num.apply(lambda t: math.pow(t, 2)).agg('sum'))
data['row_browse_num'] = data['browse_num'].apply(lambda t: t / sqrt_square_sum)


# 离散变量做特征工程
most_frequent_top10_brand_id = np.array(Counter(data.brand_id).most_common(10))[:, 0]

# 如果brand_id不是最频繁的10个brand_id,那么就给定一个默认值0，减少one-hot编码的维度
data['brand_id'] = data['brand_id'].apply(lambda t: t if t in most_frequent_top10_brand_id else 0)

# color：颜色，离散型，一共140个值，有脏数据，混入了城市，时间值，还有URL。取前10个，后面的合并。
most_frequent_top10_color = np.array(Counter(data.color).most_common(10))[:, 0]
# 如果color不是最频繁的10个color,那么就给定一个默认值0，减少one-hot编码的维度
data['color'] = data['color'].apply(lambda t: t if t in most_frequent_top10_color else '其他')

# displacement：离散特征. 排量，38053个空值。
# data.displacement.value_counts()[0:10] 取出现最多的10个。
most_frequent_top10_displacement = list(data.displacement.value_counts()[0:10].index)
data['displacement'] = data['displacement'].apply(lambda t: t if t in most_frequent_top10_displacement else '其他')

one_hot = OneHotEncoder(handle_unknown='ignore')

one_hot_data = data[['brand_id', 'color', 'displacement', 'user_province', 'user_status', ]]

one_hot.fit(one_hot_data)

# print(one_hot.categories_)
# print(one_hot.get_feature_names())

feature_array = one_hot.transform(np.array(one_hot_data)).toarray()
# 两个ndarray水平合并，跟data['id']合并，方便后面两个DataFrame合并
feature_array_add_id = np.hstack((np.asarray([data['user_id'].values]).T,
                                  np.asarray([data['car_id'].values]).T, feature_array))
# one_hot_features_df = DataFrame(feature_array, columns=one_hot.get_feature_names())
one_hot_features_df = DataFrame(feature_array_add_id,
                                columns=np.hstack((np.asarray(['user_id']), np.asarray(['car_id']), one_hot.get_feature_names())))
# 将id这一列转化为int。
one_hot_features_df['user_id'] = one_hot_features_df['user_id'].apply(lambda t: int(t))
one_hot_features_df['car_id'] = one_hot_features_df['car_id'].apply(lambda t: int(t))


# tag_ids:车辆特征标签，一个车有多个值。可以用n-hot编码。是重要的特征。108111个空值。只有3万个非空的值。
# 虽然空值有点多，但是这个特征重要，空值，可以单独编码。

# tag_ids：针对tag_ids为空的，用 '0' 替代。
data['tag_ids'] = data['tag_ids'].apply(lambda t: '0' if str(t) == 'nan' else t)
tag_ids_list = data.tag_ids.values.tolist()
tag_ids_set = set()
for x in tag_ids_list:
    tag_ids_set = tag_ids_set.union(set(x.split(',')))

# tag_ids_set = {'1', '4', '5', '2', '8', '6', '0', '7', '3'}
n_hot = OneHotEncoder(handle_unknown='ignore')
# n_hot.fit([['1'], ['4'], ['5'], ['2'], ['8'], ['6'], ['0'], ['7'], ['3']])
n_hot.fit([[x] for x in list(tag_ids_set)])

dict_vec = {}
for x in list(tag_ids_set):
    # n_hot.transform([['7']]).toarray()
    dict_vec[x] = n_hot.transform([[x]]).toarray()


# t = '1,2'
def vec(t):
    a = t.split(',')
    v = dict_vec[a[0]]
    if len(a) > 1:
        for s in a[1:]:
            v = v + dict_vec[s]
    return v


tmp = data['tag_ids'].apply(vec).values
tag_ids_array = np.asarray([list(tmp[i][0]) for i in range(tmp.size)])
tag_ids_array_add_id = np.hstack((np.asarray([data['user_id'].values]).T,
                                  np.asarray([data['car_id'].values]).T, tag_ids_array))
# n_hot_features_df = DataFrame(tag_ids_array, columns=n_hot.get_feature_names())
n_hot_features_df = DataFrame(tag_ids_array_add_id, columns=np.hstack((np.asarray(['user_id']),
                                                                       np.asarray(['car_id']), n_hot.get_feature_names())))
n_hot_features_df['user_id'] = n_hot_features_df['user_id'].apply(lambda t: int(t))
n_hot_features_df['car_id'] = n_hot_features_df['car_id'].apply(lambda t: int(t))

# 三类特征合并。
data_and_features_df = data.merge(one_hot_features_df, on=['user_id', 'car_id'],
                                  how='left').merge(n_hot_features_df, on=['user_id', 'car_id'], how='left')

user_car_features_df = data_and_features_df.drop(columns=['color', 'displacement', 'tag_ids',
                                                          'mileage', 'sell_price', 'transfer_num', 'browse_num',
                                                          'license_time', 'user_province', 'user_status', ])


# 构建用户登录相关特征

login_count = df_login.user_id.value_counts()  # Series
login_count_array = np.hstack((np.asarray([login_count.index]).T, np.asarray([login_count.values]).T,))
login_df = DataFrame(login_count_array, columns=np.hstack((np.asarray(['user_id']),
                                                           np.asarray(['login_num']),)))

login_df['log_login_num'] = login_df['login_num'].apply(lambda t: math.log(t + 1))

# login_df: ["user_id", "log_login_num"]
login_df = login_df.drop(columns=['login_num'])

# 构建模型label，利用hema_car_purchase_info和csb_car_event来构建label.
# 1. hema_car_purchase_info这个表根据brand_id、model_id、series_id能匹配到(匹配hema_car这个表)的用户和车，认为是用户喜欢的，
# 用作训练的正样本。
# 2. csb_car_event 这个表浏览二手车源表，根据浏览时长来确定正负样本。这个表中的stay_time值都是非常大的，
# 也不知道到底单位是什么，这里只要stay_time非空就认为用户喜欢该车，null就认为用户不喜欢，剔除掉car_id为空的，
# 最终还有98729条记录，其中label = 1的有24882条，label = 0的有73847条。

# astype(np.int)转化为int，原来是float
brand_id_array = df_purchase.brand_id.values.astype(np.int)
series_id_array = df_purchase.series_id.values.astype(np.int)
model_id_array = df_purchase.model_id.values.astype(np.int)

stay_df = df_event[["user_id", "car_id", "stay_time", ]]
stay_df["label"] = stay_df['stay_time'].apply(lambda t: 0 if np.isnan(t) else 1)

# df_label: ["user_id","car_id","label"]
label_df = stay_df.drop(columns=["stay_time"]).dropna(axis=0, how='any')

"""
特征拼接在一起，形成最终的训练样本空间。
车相关特征：car_feature
用户相关特征：user_one_hot_features_df
用户登录特征：login_df
label: df_label
"""

data_and_features_df = user_car_features_df.merge(login_df, on='user_id', how='left').\
    merge(label_df, on=['user_id', 'car_id'], how='left')


# brand_id_array、series_id_array、model_id_array中三个可以跟data_and_features_df中的brand_id、
# series_id、model_id匹配得上，就认为用户喜欢该车，当做模型的正样本。
# 下面的参数brand_id, series_id, model_id，都是字符串。
def like_match(brand_id, series_id, model_id, label):
    if label == 1:
        return 1
    elif np.isnan(label):
        return np.nan
    else:
        if brand_id in brand_id_array and series_id in series_id_array and model_id in model_id_array:
            return 1
        else:
            return 0


# 下面apply里面的row 对应的是DataFrame中的一行，是按照行迭代的。
data_and_features_df["label"] = data_and_features_df.\
    apply(lambda row: like_match(row['brand_id'], row['series_id'], row['model_id'], row['label']), axis=1)

# data_and_features_df.label.value_counts()
# label = 0   12561
# label =1     5609


data_and_features_df = data_and_features_df.dropna(axis=0, how='any')


# index = 0 写入时不保留索引列。
data_and_features_df.to_csv(data_path + '/' + r'logistic_model_data.csv', index=0)
# read
# data_and_features_df = pd.read_csv(data_path + '/' + r'logistic_model_data.csv')


# 展示不同的调用方式
logistic_train_df, logistic_test_df = train_test_split(data_and_features_df,
                                                       test_size=0.3, random_state=42)

logistic_train_df.to_csv(data_path + '/' + r'logistic_train_data.csv', index=0)
logistic_test_df.to_csv(data_path + '/' + r'logistic_test_data.csv', index=0)

