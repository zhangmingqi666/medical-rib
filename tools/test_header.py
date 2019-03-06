



folder_path='../data/nii_files_merges/'
import os, re
import nibabel as nib
pattern = re.compile('^[a-zA-Z1-9].*nii')
for f in os.listdir(folder_path):
    next_dir = os.path.join(folder_path, f)
    if not os.path.isdir(next_dir):
        continue

    print("read folder {}".format(f))
    for file_name in os.listdir(next_dir):

        next_next_dir = os.path.join(next_dir, file_name)
        if pattern.search(file_name) is None:
            continue
        img = nib.load(next_next_dir)
        header = img.header
        pixel_zoom = header.get_zooms()
        print("read nii {}".format(file_name), pixel_zoom)