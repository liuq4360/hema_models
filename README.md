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