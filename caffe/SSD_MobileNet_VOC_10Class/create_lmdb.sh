#!/bin/bash

# Creat lmdb for training and testing.
# Two lmdbs are going to be created under ./data/VOCdevkit/lmdb/trainval_lmdb and ./data/VOCdevkit/lmdb/test_lmdb

cur_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )


echo $CAFFE_PATH
export PYTHONPATH=$PYTHONPATH:$CAFFE_PATH/python
echo $PYTHONPATH

##Filter out annotations without defined 10 classes
echo "run voc_filter.py"
python voc_filter.py

##Generate file list for annotation conversion
echo "generate file list"
python generate_file_list.py

##Create lmdb
echo "creating lmdb"
redo=1
data_root_dir="./data/VOCdevkit"
mapfile="labelmap_voc.prototxt"
anno_type="detection"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0

extra_cmd="--encode-type=jpg --encoded"
if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi
for subset in test trainval
do
  python $CAFFE_PATH/scripts/create_annoset.py --anno-type=$anno_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir data/$subset.txt $data_root_dir/$db/"$subset"_"$db" ./
done
