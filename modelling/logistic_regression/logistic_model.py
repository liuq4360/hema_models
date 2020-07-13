from sklearn.linear_model import LogisticRegression
import pandas as pd
import csv
import numpy as np
import math
from pandas import Series, DataFrame
import os
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay

"""
该脚本主要完成3件事情：
1. 训练logistic回归模型；
2. 针对测试集进行预测；
3. 评估训练好的模型在测试集上的效果；

这个脚本中的所有操作都可以借助scikit-learn中的函数来实现，非常简单。
这里为了简单起见，将模型训练、预测与评估都放在这个文件中了。



关于logistic回归模型各个参数的含义及例子可以参考，https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression

关于模型评估的案例可以参考：https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_display_object_visualization.html#sphx-glr-auto-examples-miscellaneous-plot-display-object-visualization-py

"""

cwd = os.getcwd()  # 获取当前工作目录
data_path = os.path.abspath(os.path.join(cwd, "../../data"))  # 获取data目录

logistic_train_df = pd.read_csv(data_path + '/' + r'logistic_train_data.csv')


"""
下面代码是训练logistic回归模型。
"""
clf = LogisticRegression(penalty='l2',
                         solver='liblinear', tol=1e-6, max_iter=1000)

X_train = logistic_train_df.drop(columns=['user_id', 'car_id',
                                          'brand_id', 'series_id', 'model_id', 'label', ])
y_train = logistic_train_df['label']

clf.fit(X_train, y_train)

# clf.coef_
# clf.intercept_
# clf.classes_

"""
下面的代码用上面训练好的logistic回归模型来对测试集进行预测。
"""
logistic_test_df = pd.read_csv(data_path + '/' + r'logistic_test_data.csv')

X_test = logistic_test_df.drop(columns=['user_id', 'car_id',
                                        'brand_id', 'series_id', 'model_id', 'label', ])

y_test = logistic_test_df['label']

# logistic回归模型预测出的结果为y_score
y_score = clf.predict(X_test)

# 包含概率值的预测
# y_score = clf.predict_proba(X_test)

# np.unique(Z)
# Counter(Z).most_common(2)
# logistic_test_df.label.value_counts()


"""
下面的代码对logistic回归模型进行效果评估，主要有3种常用的评估方法：
1. 混淆矩阵：confusion matrix
2. roc曲线：roc curve
3. 精准度和召回率：precision recall
"""

# confusion matrix
y_score = clf.predict(X_test)
cm = confusion_matrix(y_test, y_score)
cm_display = ConfusionMatrixDisplay(cm).plot()

# roc curve
fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=clf.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

# precision recall
prec, recall, _ = precision_recall_curve(y_test, y_score, pos_label=clf.classes_[1])
pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()

