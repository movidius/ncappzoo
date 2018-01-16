#!/usr/bin/env sh
# Create the dogsvscats lmdb inputs
# N.B. set the path to the dogsvscats train + val data dirs
# Adopted from CAFFE/examples/imagenet/create_imagenet.sh

set -e

DATA_ROOT=$1
DATA=$DATA_ROOT/train/
TOOLS=$CAFFE_PATH/build/tools/

TRAIN_DATA_ROOT=$DATA
VAL_DATA_ROOT=$DATA

TRAIN_LMDB_ROOT=$DATA_ROOT
VAL_LMDB_ROOT=$DATA_ROOT

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=224
  RESIZE_WIDTH=224
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_dogsvscats.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_dogsvscats.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA/train.txt \
    $TRAIN_LMDB_ROOT/dogsvscats_train_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_ROOT \
    $DATA/val.txt \
    $VAL_LMDB_ROOT/dogsvscats_val_lmdb

echo "Done."
