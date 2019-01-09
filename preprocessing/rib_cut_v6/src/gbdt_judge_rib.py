import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO
from  sklearn import tree
import pydotplus
import pickle as pkl

bone_data = pd.read_csv(open('/home/jiangyy/projects/medical-rib/bone_df_csv/all_bone_info_df.csv'))

# print(bone_data[bone_data.isnull().values==True])

x = bone_data[['z_centroid/z_shape', 'x_centroid/x_shape', 'y_centroid/y_shape', 'z_max/z_shape', 'x_max/x_shape',
               'y_max/y_shape', 'z_min/z_shape', 'x_min/x_shape', 'y_min/y_shape', 'z_length/z_shape', 'x_length/x_shape',
               'y_length/y_shape', 'iou_on_xoy', 'distance_nearest_centroid', 'point_count', 'mean_z_distance_on_xoy', 'max_nonzero_internal']]

y = bone_data[['is_rib']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

gbdt = GradientBoostingClassifier(random_state=10)
gbdt.fit(x_train, y_train)

joblib.dump(gbdt, '/home/jiangyy/projects/medical-rib/model/gbdt.pkl')

y_pred = gbdt.predict(x_test)
print(y_pred)
# y_predprob = gbdt.predict_proba(x_train)[:, 1]
print("accuracy: %.4g" % (metrics.accuracy_score(y_test, y_pred)))

print(gbdt.n_features)
print(gbdt.feature_importances_)

"""decision tree
dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(x_train, y_train)
print(dtc.score(x_test, y_test))
print(dtc.feature_importances_)
print(dtc.classes_)

dot_data = StringIO()
tree.export_graphviz(dtc, out_file=dot_data, feature_names=['z_centroid/z_shape', 'x_centroid/x_shape', 'y_centroid/y_shape', 'z_max/z_shape', 'x_max/x_shape',
               'y_max/y_shape', 'z_min/z_shape', 'x_min/x_shape', 'y_min/y_shape', 'z_length/z_shape', 'x_length/x_shape',
               'y_length/y_shape', 'iou_on_xoy', 'distance_nearest_centroid', 'point_count', 'mean_z_distance_on_xoy', 'max_nonzero_internal'], class_names=['0', '1'], filled=True,
                     rounded=True, impurity=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('viz.pdf')
"""
