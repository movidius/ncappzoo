#!/bin/sh
input_path=$1
ckp=`ls -t "$input_path"/model.ckpt-*.meta | head -1`
ckp="${ckp/.meta/}"
echo $ckp