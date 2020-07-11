# hema_models

一、KMeans模型

1. 数据预处理
程序：/preprocessing/kmeans/data_process.py

input： 输入的数据是 /data/hema_car-new.csv

功能：筛选特征，将特征转化为数值向量。

output：结果存储在 /data/train.csv



2. kmeans聚类

程序：/modelling/kmeans/car_clustering.py

input： 输入的数据是 /data/train.csv

功能：对第1步构建的车的特征向量进行聚类。

output：结果存储在 /data/kmeans.csv



3. 为用户推荐喜欢的车

程序：/infer/kmeans/user_rec.py

input： 输入的数据是 /data/kmeans.csv + /data/hema_car-new.csv

功能：基于用户最近喜欢的车，给他推荐可能喜欢的车。

output：输出直接打印出，没有存储到文件中。



二、logistic回归模型

0. 数据集描述

(1) hema_car：二手车表，车辆相关信息。用于获取与车相关的特征。
一共 133972条记录。
id：车辆id，一共有130634条。
user_id：用户id，一共有5884个。

该表中既包含车相关信息也包含人相关信息，都可以作为特征。

(2) csb_car_event：浏览二手车源表。用户行为记录。浏览代表了用户的兴趣。
一共372306条记录。car_id非空的有98729条。
car_id：24900个。
user_id：8939个。

(3) hema_car_purchase_info：线上求购表。求购代表了用户的兴趣。
一共1939条记录。
brand_id：88个
series_id：404个
model_id：1014个
user_id：176个
这个表没有car_id，可以根据brand_id、series_id、model_id来关联，这
个表中的每条记录也代表了用户的一种兴趣。

(4) hema_user_login_record：用户登录记录表。
一共251834条记录。
user_id：48461个。
用户登录的频次可以作为一个特征。

由于整个数据比较乱，很多表对该项目没有价值，该项目只用上面4个表来
构建logistic回归模型。