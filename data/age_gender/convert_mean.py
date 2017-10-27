#! /usr/bin/env python3

import caffe
import numpy as np
import sys
import os

blob = caffe.proto.caffe_pb2.BlobProto()
data = open( 'mean.binaryproto', 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
out = arr[0]
np.save( 'age_gender_mean.npy', out )

