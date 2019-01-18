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
import warnings

warnings.filterwarnings("ignore")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Search some files')

    parser.add_argument('--dataset_path', required=True, dest='dataset_path', action='store', help='dataset_path')
    parser.add_argument('--saved_gbdt_path', required=True, dest='saved_gbdt_path', action='store', help='saved_gbdt_path')
    parser.add_argument('--saved_feature_path', required=True, dest='saved_feature_path', action='store', help='saved_feature_path')
    args = parser.parse_args()

    bone_data = pd.read_csv(open(args.dataset_path))
    features_list = list(bone_data.columns)[1:]
    features_list.remove('class_num')
    features_list.remove('target')

    x = bone_data[features_list]
    y = bone_data[['target']]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    gbdt = GradientBoostingClassifier(random_state=3)
    gbdt.fit(x_train, y_train)

    # sava gbdt model and feature list
    joblib.dump(gbdt, args.saved_gbdt_path)
    joblib.dump(features_list, args.saved_feature_path)

    y_pred = gbdt.predict(x_test)
    print(y_pred)
    print("accuracy: %.4g" % (metrics.accuracy_score(y_test, y_pred)))
    print(features_list)
    print(gbdt.n_features)
    print(gbdt.feature_importances_)


if __name__ == '__main__':
    main()
