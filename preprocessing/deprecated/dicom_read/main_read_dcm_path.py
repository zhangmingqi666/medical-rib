from Dcm_Info import *
import os


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Search some files')
    parser.add_argument('--dcm_folder', dest='dcm_folder', action='store', help='dcm_path', default=None)
    parser.add_argument('--dcm_file_csv_path', dest='dcm_file_csv_path', action='store', help='dcm_path', default=None)

    args = parser.parse_args()

    if os.path.exists(args.dcm_folder) and (args.dcm_file_csv_path is not None):
        dcm_df = find_all_dcm_info_and_its_id(path=args.dcm_folder)
        dcm_df.to_csv(args.dcm_file_csv_path, index=False)


if __name__ == '__main__':
    main()
