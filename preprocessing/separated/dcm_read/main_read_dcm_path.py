from preprocessing.separated.dcm_read import Dcm_Info


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Search some files')
    parser.add_argument('--dcm_folder', nargs='+', dest='dcm_folder', action='store')
    # parser.add_argument('--dcm_folder', dest='dcm_folder', action='store', help='dcm_path', default=None)
    parser.add_argument('--dcm_file_csv_path', dest='dcm_file_csv_path', action='store', help='dcm_path', default=None)

    args = parser.parse_args()

    dcm_df = Dcm_Info.find_all_dcm_info_and_its_id(paths=args.dcm_folder)
    dcm_df.to_csv(args.dcm_file_csv_path, index=False)


if __name__ == '__main__':
    main()
