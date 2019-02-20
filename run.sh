

cd preprocessing/dicom_read
./pretreat.sh ../../dataSet/updated48labeled_1.31

cd -
cd preprocessing/rib_cut_v7
./cut_ribs.sh ../../dataSet/updated48labeled_1.31
