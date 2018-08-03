#!/bin/bash

#Check if net prototxt exists
net_dir="mobilenet-ssd-1.0-x10"
if [ -d $CAFFE_PATH$net_dir ]
then
   if [ ! -d $net_dir ]
   then 
      ln -s $CAFFE_PATH$net_dir
   fi  
else
   echo "Make sure CAFFE_PATH set in env"
   echo "Make sure clone caffe from https://github.com/yuanyuanli85/caffe.git ssd_x10 branch"
fi

#create snapshot
if [ ! -d $net_dir"/snapshot" ]
then
  mkdir $net_dir"/snapshot"
fi 

#create soft link for lmdb 
trainval_lmdb=$PWD"/data/VOCdevkit/lmdb/trainval_lmdb"
test_lmdb=$PWD"/data/VOCdevkit/lmdb/test_lmdb"
ln -s $trainval_lmdb $net_dir"/trainval_lmdb"
ln -s $test_lmdb $net_dir"/test_lmdb"

#start training
cd $net_dir
$CAFFE_PATH/build/tools/caffe train -solver='solver.prototxt' -weights='mobilenet_1.0_feature.caffemodel' -gpu 0

