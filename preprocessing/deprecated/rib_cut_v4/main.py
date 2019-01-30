#coding=utf-8
#from .src.Bone_decompose import decompose
from src.Dicom_Read import *
from src.Bone_decompose import *
import os
import pickle
import argparse


def main():

    parser = argparse.ArgumentParser(description='Search some files')

    parser.add_argument('--pkl_path', dest='pkl_path', action='store', help='pkl_path', default=None)
    parser.add_argument('--dcm_path', dest='dcm_path', action='store', help='dcm_path', default=None)
    parser.add_argument('--output_prefix',  dest='output_prefix', action='store', help='prefix', default='prefix')
    args = parser.parse_args()

    rib_data = None
    if args.pkl_path is not None:
        print("the rib data created from pkl_path {}.".format(args.pkl_path))
        rib_data = pickle.load(open(args.pkl_path, 'rb'))
    elif args.dcm_path is not None:
        print("the rib data created from pkl_path {}.".format(args.dcm_path))
        rib_data = RibDataFrame().readDicom(path=args.dcm_path)
    else:
        print('cannot create rib_data by providing path.')
        exit(1)

    main_excute(rib_data, allow_debug=True, output_prefix=args.output_prefix)

    #remove = decompose.remove(rib_data)
    #result = remove.separate_Bones()


if __name__ == '__main__':
    main()