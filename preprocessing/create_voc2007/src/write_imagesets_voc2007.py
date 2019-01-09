import os
import random
import argparse
import numpy as np
import sys


def get_all_data(xml_path=None):
    all_xml = os.listdir(xml_path)
    res = [e.replace('.xml', '') for e in all_xml]
    return res


def generate_train_val_test(data_list=None, sample_prob=[0.6, 0.15, 0.25], main_path=None):
    index = np.random.choice(len(sample_prob), len(data_list), p=sample_prob)
    data_arr = np.array(data_list)
    train_arr = data_arr[index == 0]
    val_arr = data_arr[index == 1]
    test_arr = data_arr[index == 2]

    assert os.path.exists(main_path)
    with open(os.path.join(main_path, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_arr.tolist()))

    with open(os.path.join(main_path, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_arr.tolist()))

    with open(os.path.join(main_path, 'test.txt'), 'w') as f:
        f.write('\n'.join(test_arr.tolist()))

    with open(os.path.join(main_path, 'trainval.txt'), 'w') as f:
        f.write('\n'.join(train_arr.tolist() + val_arr.tolist()))


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Search some files')
    parser.add_argument('--voc2007_Annotations_folder', required=True, dest='voc2007_Annotations_folder',
                        action='store', help='voc2007_Annotations_folder')
    parser.add_argument('--voc2007_ImageSets_folder', required=True,
                        dest='voc2007_ImageSets_folder', action='store',
                        help='voc2007_ImageSets_folder')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if not os.path.exists(args.voc2007_Annotations_folder) or not os.path.exists(args.voc2007_ImageSets_folder):
        print('voc2007_Annotations_folder or voc2007_ImageSets_folder not exist')
        exit(1)

    data_list = get_all_data(xml_path=args.voc2007_Annotations_folder)
    generate_train_val_test(data_list=data_list, sample_prob=[0.6, 0.15, 0.25], main_path=args.voc2007_ImageSets_folder)




