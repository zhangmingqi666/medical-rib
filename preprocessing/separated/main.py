#coding=utf-8
import pickle
import argparse
import sys, os


def add_python_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


add_python_path(os.getcwd())
# from projects
from preprocessing.separated.ribs_obtain.Bone_decompose import void_cut_ribs_process
from preprocessing.separated.dcm_read import Dicom_Read


def main():

    parser = argparse.ArgumentParser(description='Search some files')
    parser.add_argument('--use_pkl_or_dcm', dest='use_pkl_or_dcm', action='store',
                        type=str, help='use_pkl_or_dcm', default='dcm')
    parser.add_argument('--dcm_path', dest='dcm_path', action='store', help='dcm_path', default=None)
    parser.add_argument('--keep_slicing', dest='keep_slicing', action='store', type=int,
                        help='keep_slicing', default=False)
    parser.add_argument('--pkl_path', dest='pkl_path', action='store', help='pkl_path', default=None)
    parser.add_argument('--output_prefix',  dest='output_prefix', action='store', help='prefix', default=None)
    parser.add_argument('--rib_df_cache_path',  dest='rib_df_cache_path', action='store',
                        help='rib_df_cache_path', default=None)
    parser.add_argument('--bone_info_path', dest='bone_info_path', action='store',
                        help='bone_info_path', default=None)
    parser.add_argument('--rib_recognition_model_path',  dest='rib_recognition_model_path',
                        action='store', help='rib_recognition_model_path', default=None)

    args = parser.parse_args()

    keep_slicing = True if args.keep_slicing != 0 else False
    print("input data format is {}".format(args.use_pkl_or_dcm), type(args.use_pkl_or_dcm))
    print("keep_slicing is {}".format(keep_slicing))
    print("the dcm path is {}".format(args.dcm_path))

    rib_data = None
    if args.use_pkl_or_dcm == 'pkl' and args.pkl_path is not None:
        print("the rib data created from pkl_path {}.".format(args.pkl_path))
        rib_data = pickle.load(open(args.pkl_path, 'rb'))
        print("3d image shape is {}".format(rib_data.shape))
    elif args.use_pkl_or_dcm == 'dcm' and os.path.isdir(args.dcm_path):
        print("the rib data created from pkl_path {}.".format(args.dcm_path))
        rib_data, new_spacing = Dicom_Read.RibDataFrame().readDicom(path=args.dcm_path, keep_slicing=keep_slicing)
        pickle.dump(rib_data, open(args.pkl_path, 'wb'))
        print("3d image shape is {}, new_spacing is {}".format(rib_data.shape, new_spacing))
    else:
        print('cannot create rib_data by providing path.')
        exit(1)

    void_cut_ribs_process(rib_data, allow_debug=True, output_prefix=args.output_prefix, 
                          rib_df_cache_path=args.rib_df_cache_path, bone_info_path=args.bone_info_path,
                          rib_recognition_model_path=args.rib_recognition_model_path)


if __name__ == '__main__':
    main()
