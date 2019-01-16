# medical-rib
<script src="/js/mermaid.full.min.js">
graph TB
	subgraph Preprocessing
        subgraph Extract Rib Model
            subgraph samples
                A[source dicom file]
            end
            A --> |dicom_read| B(CT pkl file)
            B --> |rib_cut_and_extract| C(rib_df_cache)
        end

        C --> |offset| G(offset_df.csv)

        subgraph labels
            D[.nii files]
            E[patient_info_excel]
        end

        D --> F{label_info.csv}
        E --> F
        F --> H{data_join_label.csv}
        C --> H

        I{label_loc_type_info.csv}
        G --> I
        F --> I
        H --> I
    end
    I --> J(VOC 2007 xml)
    subgraph Fracture Detection Model
        J --> K(faster r-cnn)
        style K fill:#f9f,stroke:#333,stroke-width:4px
        K --> L(output of detecting fracture location)
    end
</script>

<script src="/js/mermaid.full.min.js">
graph TB
    subgraph Extract Rib Model
    subgraph samples
    A[source dicom file]
    end
    A --> |dicom_read| B(CT pkl file)
    B --> |rib_cut_and_extract| C(rib_df_cache)
    end
    
    C --> |offset| G(offset_df.csv)
    
    subgraph labels
    D[.nii files]
    E[patient_info_excel]
    end
    
    D --> F{label_info.csv}
    E --> F
    F --> H{data_join_label.csv}
    C --> H
    
    I{label_loc_type_info.csv}
    G --> I
    F --> I
    H --> I
    I --> J(VOC 2007 xml)

    subgraph Fracture Detection Model
    J --> K(faster r-cnn)
    style K fill:#f9f,stroke:#333,stroke-width:4px
    K --> L(output of detecting fracture location)
    end
</script>
