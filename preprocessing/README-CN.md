
### 数据集结构

我们的原数据有来自不同医院的多个批次, 因此我们一开始就严格定义文件结构和数据格式

```
data
└───csv_files
│       offset_df.csv
│       rib_type_location.xls    
│       nii_loc_df.csv
│       label_loc_type_info.csv    
│       join_label.csv
│ 
└───dicom_files_merges
│   └───patient001
│   │       file111.dcm
│   │       file112.dcm
│   │       ...
│   │
│   └───patient002
│           ...
│
└───nii_files_merges
│   └───patient001
│   │       file111.nii
│   │       file112.nii
│   │       ...
│   └───patient002
│           ...
│
└───ribs_df_cache
│       patient001.csv
│       patient002.csv
│       ...
│
└───bone_info_merges
│       patient001.csv
│       patient002.csv
|       ...
|
└───voc2007
        train.txt
        2007_test.txt
        Annotations
        ImageSets
        JPEGImages
        labels
```

`dicom_files_merges`, `nii_files_merges`, `csv_files/rib_type_location.xls`文件/文件夹由医院提供, 剩下的由我们算法生成, 以下是更详细的介绍

+ **dicom_files_merges**: 存储CT图像, 每个文件夹代表一个病人,
+ **nii_files_merges**: 受伤位置被标注, 每个.nii文件表示肋骨上的一个和多个受伤位置,
+ **bone_info_merges**: 记录每个骨骼的特征和预测结果(是否肋骨), 更新预测错误的类标并合并成训练集,
+ **rib_df_cache**: 某个病人的肋骨收集存储在`patient_id.csv`, 列名有`x,y,z,c,v`(c是肋骨id, v是hu),
+ **csv_files/rib_type_location.xls**: 存储nii文件中受伤位置对应的骨折类型等,
+ **csv_files/nii_loc_df.csv**: 存储每个受伤位置的3D bounding box,
+ **csv_files/join_label.csv**: 受伤位置和肋骨的连接,
+ **csv_files/offset_df.csv**: 肋骨偏移

如果需要扩大数据集, 参考[数据标注方法](./DATA_ANNOTATION_METHODS.md)

### dataflow

```
graph TB
    subgraph preprocessing
        subgraph provided by hospitals
            A(dicom_files_merges)
            D(nii_files_merges)
            E(csv_files/rib_type_location.xls)
        end
        
        subgraph ribs extraction
            A --> |ribs_obtain.sh| B(ribs_cache_df/patient_id.csv)
        end

        subgraph match
            B --> |prepare_data.sh|H[csv_files/data_join_label.csv,ribs picture,locations]
            D --> |nii_read.sh|F[csv_files/nii_loc_df.csv]
            E --> |prepare_data.sh|H
            F --> |prepare_data.sh|H
            H --> |prepare_data.sh|J(voc2007 format)
        end
    end

    subgraph Fracture Detection
        J ==train==> K(yolo-v3,using darknet)
        style K fill:#f9f,stroke:#333,stroke-width:4px
        B -->|predict,demo.sh|K
    end
    
    K -->|predict,demo.sh|L(predict scores)
```

![flowchart](.github/flow_chart.png)
