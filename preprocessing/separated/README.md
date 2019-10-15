

### dcm read

slices of CT data maybe saved to everywhere in a root path. we need to find the absolute path using dfs method and save `id,abs_path` in `csv_files/dicom_info.csv`

### nii read

nii files will be annotated and provided by professionals, which has same shape with CT data. 
some sparse nonzero in the 3D matrix stand for fragmented locations. we labeled connected regions using morphology and get the 3D bounding box of 
every connected regions.

### ribs detection

+ separate bones using morphology openings
+ remove the spine and the sternum
+ collected all the remaining bones
+ recognize and collect ribs, save `ribs_cache_df/id.csv`


