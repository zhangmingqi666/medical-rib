

### dcm 读取

CT slices 有可能存储在某个主目录的任何分支,我们需要使用dfs找到其绝对路径并将`id,abs_path`存储`csv_files/dicom_info.csv`文件中

### nii 读取

nii文件由医院的专家标定和提供, nii文件和CT数据shape一致. nii文件中的非0点是骨折位置.我们使用形态学标定其中的相连区域并且获得各个相连区域的bounding box

### 肋骨识别

+ 使用形态学方法分离骨头
+ 移除脊柱和胸骨
+ 收集剩余骨头
+ 识别其中的肋骨并存储在`ribs_cache_df/id.csv`文件中


