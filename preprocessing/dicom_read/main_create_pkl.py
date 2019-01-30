from Dicom_Read import RibDataFrame
import os
import pickle
import warnings
warnings.filterwarnings('ignore')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Search some files')

    parser.add_argument('--dcm_path', required=True, dest='dcm_path', action='store', help='dcm_path')
    parser.add_argument('--output_pkl_path', required=True, dest='output_pkl_path', action='store', help='dcm_path')
    args = parser.parse_args()

    if not os.path.exists(args.dcm_path):
        print('dcm_path dirs not exist')
        exit(1)

    pix_resampled = RibDataFrame().readDicom(path=args.dcm_path)

    pickle.dump(pix_resampled, open(args.output_pkl_path, "wb"))


if __name__ == '__main__':
    main()
