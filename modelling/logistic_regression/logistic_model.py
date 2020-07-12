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


cwd = os.getcwd()  # 获取当前工作目录
data_path = os.path.abspath(os.path.join(cwd, "../../data"))  # 获取data目录

logistic_train_df = pd.read_csv(data_path + '/' + r'logistic_train_data.csv')


clf = LogisticRegression(penalty='l2',
                         solver='liblinear', tol=1e-6, max_iter=1000)

X_train = logistic_train_df.drop(columns=['user_id', 'car_id',
                                          'brand_id', 'series_id', 'model_id', 'label', ])
y_train = logistic_train_df['label']

clf.fit(X_train, y_train)

# clf.coef_
# clf.intercept_
# clf.classes_

logistic_test_df = pd.read_csv(data_path + '/' + r'logistic_test_data.csv')

X_test = logistic_test_df.drop(columns=['user_id', 'car_id',
                                        'brand_id', 'series_id', 'model_id', 'label', ])

y_test = logistic_test_df['label']

y_score = clf.predict(X_test)

# ZZ = clf.predict_proba(logistic_test_df.drop(columns=['user_id', 'car_id',
#                                                       'brand_id', 'series_id', 'model_id', 'label', ]))

# np.unique(Z)
# Counter(Z).most_common(2)
# logistic_test_df.label.value_counts()


"""
logistic 回归效果评估
1. confusion matrix
2. roc curve
3. precision recall
"""

# confusion matrix
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()

# roc curve
fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=clf.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

# precision recall
prec, recall, _ = precision_recall_curve(y_test, y_score,
                                         pos_label=clf.classes_[1])
pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()

