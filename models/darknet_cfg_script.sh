
git clone https://github.com/AlexeyAB/darknet.git
cd darknet
make
wget https://pjreddie.com/media/files/darknet53.conv.74
cd -
cp ./darknet_cfg/Makefile ./darknet/
cp ./darknet_cfg/data/* ./darknet/data/
cp ./darknet_cfg/cfg/* ./darknet/cfg/
