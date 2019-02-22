from Dicom_Read import RibDataFrame
import os
import pickle
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Search some files')

    parser.add_argument('--dcm_df_path', required=True, dest='dcm_df_path', action='store', help='dcm_df_path')
    parser.add_argument('--output_pkl_folder', required=True, dest='output_pkl_folder', action='store',
                        help='output_pkl_folder')
    parser.add_argument('--spacing_df_path', required=True, dest='spacing_df_path', action='store',
                        help='spacing_df_path')
    args = parser.parse_args()

    dcm_df = pd.read_csv(args.dcm_df_path)
    spacing_df = pd.DataFrame({}, columns=['id', 'spacing0'])
    for index, row in dcm_df.iterrows():
        id, dcm_path = row['id'], row['dcm_path']
        output_pkl_path = "{}/{}.pkl".format(args.output_pkl_folder, id)
        print("start make {}".format(output_pkl_path))
        pix_resampled, spacing0 = RibDataFrame().readDicom(path=dcm_path)
        pix_resampled = pix_resampled.astype(np.int16)
        pickle.dump(pix_resampled, open(output_pkl_path, "wb"))
        spacing_df.loc[len(spacing_df)] = {'id': id, 'spacing0': spacing0}
    spacing_df.to_csv(args.spacing_df_path)


if __name__ == '__main__':
    main()

