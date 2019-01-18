"""
@Author: Leaves
@Time: 17/01/2019
@File: gbdt_judge_rib.py
@Function: train and save GBDT model
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

bone_data = pd.read_csv(open('/home/jiangyy/projects/medical-rib/bone_df_csv/all_bone_info_df.csv'))
features_list = list(bone_data.columns)[1:]
features_list.remove('label')
features_list.remove('is_rib')
features_list.remove('quantile_up_z_distance_on_xoy')
features_list.remove('quantile_down_z_distance_on_xoy')

x = bone_data[features_list]

y = bone_data[['is_rib']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

gbdt = GradientBoostingClassifier(random_state=3)
gbdt.fit(x_train, y_train)

joblib.dump(gbdt, '/home/jiangyy/projects/medical-rib/model/gbdt.pkl')
joblib.dump(features_list, '/home/jiangyy/projects/medical-rib/model/feature.pkl')

y_pred = gbdt.predict(x_test)
print(y_pred)
# y_predprob = gbdt.predict_proba(x_train)[:, 1]
print("accuracy: %.4g" % (metrics.accuracy_score(y_test, y_pred)))
print(features_list)
print(gbdt.n_features)
print(gbdt.feature_importances_)

