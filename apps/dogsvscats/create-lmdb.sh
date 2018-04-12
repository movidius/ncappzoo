#!/usr/bin/env sh
# Create the dogsvscats lmdb inputs
# N.B. set the path to the dogsvscats train + val data dirs
# Adopted from CAFFE/examples/imagenet/create_imagenet.sh

set -e

DATA_DIR=$1
TOOLS=$CAFFE_SRC_PATH/build/tools/

TRAIN_DATA_DIR=$DATA_DIR/train/
VAL_DATA_DIR=$DATA_DIR/train/

TRAIN_LMDB_DIR=$DATA_DIR/
VAL_LMDB_DIR=$DATA_DIR/

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

if [ ! -d "$TRAIN_DATA_DIR" ]; then
  echo "Error: TRAIN_DATA_DIR is not a path to a directory: $TRAIN_DATA_DIR"
  echo "Set the TRAIN_DATA_DIR variable in create-lmdb.sh to the path" \
       "where the dogsvscats training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_DIR" ]; then
  echo "Error: VAL_DATA_DIR is not a path to a directory: $VAL_DATA_DIR"
  echo "Set the VAL_DATA_DIR variable in create-lmdb.sh to the path" \
       "where the dogsvscats validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_DIR \
    $DATA_DIR/train.txt \
    $TRAIN_LMDB_DIR/train_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_DIR \
    $DATA_DIR/val.txt \
    $VAL_LMDB_DIR/val_lmdb

echo "Done."
