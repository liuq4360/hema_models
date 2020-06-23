import pandas as pd
import csv
import numpy as np
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder
import math
from pandas import Series, DataFrame
import os

"""
数据预处理，选择出可以作为特征的变量：
            'brand_id', 'color', 'displacement', 'tag_ids',
            'mileage', 'sell_price', 'transfer_num', 'browse_num',
            'license_time'
            
这些变量有3大类，类别变量、数值变量、时间变量。

类别变量通过one-hot和n-hot编码，转化为向量形式。其中one-hot编码用了scikit-learn中的OneHotEncoder方法。
n-hot编码是自己实现的。

数值变量通过适当的数据变换(即特征预处理)转化为具体的数值。

时间变量，取年份，用该年份到2020年之间间隔的年数作为数值特征。

该脚本通过DataFrame的一些操作将上面的9类特征，最终转化为45个数值特征，供后面的KMeans模型训练。

最终的特征加上一列id(id放在最前面)转化为csv文件存在data目录下的train.csv中
           
"""

cwd = os.getcwd()  # 获取当前工作目录
data_path = os.path.abspath(os.path.join(cwd, "../../data"))  # 获取data目录

# fo = open(input_path + '/' + r'hema_car-new.csv', 'w')
# ff = csv.writer(fo)
# with open(input_path + '/' + r'hema_car.csv', 'r') as f:
#     # reader = csv.reader(f)
#     reader = csv.reader((line.replace('\0', '') for line in f))
#     for i in reader:
#         ff.writerow(i)
# fo.close()

# df = pd.read_csv(input_path + '/' + r'hema_car-new.csv',
#                  dtype={'id' : np.int,
#                         'brand_id' : np.int,
#                         'color': 'category',
#                         'mileage':np.float,
#                         'sell_price':np.float,
#                         'tag_ids':np.str,
#                         'transfer_num' : np.int,
#                         'browse_num' : np.int,
#                         'displacement' : 'category'},
#                  parse_dates=['license_time'])

df = pd.read_csv(data_path + '/' + r'hema_car-new.csv')

# df.columns
#
# data = df[['id', 'brand_id']] # 取出两列
# df[df.sell_price > 400]  # 某列值大于400的所有数据

# np.nan   numpy中的空值
# df[np.isnan(df.new_car_price)] # 取出new_car_price中空值的数据
# df[df.new_car_price.isnull()]

# df.license_time.unique() # license_time 所有可能的取值
# df.mileage.nlargest(3,keep='all') # 取最大3个值
# data.sell_price.nsmallest(3,keep='all') # 取最大3个值
# data.displacement.value_counts()[0:10] 取出现最多的10个。

# 选择9个特征：
# brand_id: one-hot编码
# color: one-hot编码
# displacement：离散特征
# tag_ids：离散特征，n-hot编码

# mileage: 数值特征，
# sell_price: 数值特征，
# transfer_num：数值特征，
# browse_num：数值特征，

# license_time：时间特征，转化为数值

data = df[['id',
           'brand_id', 'color', 'displacement', 'tag_ids',
           'mileage', 'sell_price', 'transfer_num', 'browse_num',
           'license_time', ]]


# 特征工程


# license_time: 将上牌时间到现在过了多少年作为一个特征
data['past_year'] = data['license_time'].apply(lambda x: 2020 - int(str(x)[0:4]))

# mileage：最大值为172416.0，最小值为0，最大值取对数后为12. x -> log(x+1)
data['log_mileage'] = data['mileage'].apply(lambda t: math.log(t+1))

# sell_price: 去掉价格高于400万的50条，可能是脏数据
data = data[data.sell_price < 400]
# sell_price 用min-max归一化
max_sell_price = data.sell_price.nlargest(1, keep='all').values[0]
min_sell_price = data.sell_price.nsmallest(1, keep='all').values[0]
data['min_max_sell_price'] = data['sell_price'].apply(lambda t: (t - min_sell_price)/(max_sell_price - min_sell_price))

# transfer_num: 最大值22，最小值0，用正态分布归一化. 17144个空值。 用0填充。
data['transfer_num'] = data['transfer_num'].apply(lambda t: 0 if np.isnan(t) else t)
mean_transfer_num = np.mean(data.transfer_num)
std_transfer_num = np.std(data.transfer_num)
data['gauss_transfer_num'] = data['transfer_num'].apply(lambda t: (t - mean_transfer_num)/std_transfer_num)

# browse_num, 数值特征，最大值1060，最小值0，行归一化。
sqrt_square_sum = math.sqrt(data.browse_num.apply(lambda t: math.pow(t, 2)).agg('sum'))
data['row_browse_num'] = data['browse_num'].apply(lambda t: t/sqrt_square_sum)

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

one_hot_data = data[['brand_id', 'color', 'displacement', ]]

one_hot.fit(one_hot_data)

# print(one_hot.categories_)
# print(one_hot.get_feature_names())

feature_array = one_hot.transform(np.array(one_hot_data)).toarray()
# 两个ndarray水平合并，跟data['id']合并，方便后面两个DataFrame合并
feature_array_add_id = np.hstack((np.asarray([data['id'].values]).T, feature_array))
# one_hot_features_df = DataFrame(feature_array, columns=one_hot.get_feature_names())
one_hot_features_df = DataFrame(feature_array_add_id, columns=np.hstack((np.asarray(['id']), one_hot.get_feature_names())))
# 将id这一列转化为int。
one_hot_features_df['id'] = one_hot_features_df['id'].apply(lambda t: int(t))

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
tag_ids_array_add_id = np.hstack((np.asarray([data['id'].values]).T, tag_ids_array))
# n_hot_features_df = DataFrame(tag_ids_array, columns=n_hot.get_feature_names())
n_hot_features_df = DataFrame(tag_ids_array_add_id, columns=np.hstack((np.asarray(['id']), n_hot.get_feature_names())))
n_hot_features_df['id'] = n_hot_features_df['id'].apply(lambda t: int(t))

# 三类特征合并。
data_and_features_df = data.merge(one_hot_features_df, on='id',
                                  how='left').merge(n_hot_features_df, on='id', how='left')

df_train = data_and_features_df.drop(columns=['brand_id', 'color', 'displacement', 'tag_ids',
                                              'mileage', 'sell_price', 'transfer_num', 'browse_num',
                                              'license_time', ])

# index = 0 写入时不保留索引列。
df_train.to_csv(data_path + '/' + r'train.csv', index=0)
# read
# df = pd.read_csv(data_path + '/' + r'train.csv')
