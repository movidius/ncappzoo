#!/usr/bin/python3

# ****************************************************************************
# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.
# ****************************************************************************

# Create labels files for training & validation data 
# Usage: python create-labels.py /path/to/data/train/

import os
import sys
import glob
import ntpath

IMAGES_PATH = sys.argv[1] + '/train/'

# ---- Main function (entry point for this script ) --------------------------
def main():
    fVal = open( IMAGES_PATH + 'val.txt', 'w' )
    fTrain = open( IMAGES_PATH + 'train.txt', 'w' )

    # Create a list of all files in current directory & sub-directories
    file_list = [ os.path.basename(img) for img in 
                  glob.glob( IMAGES_PATH + '*.jpg' ) ]

    for file_index, file_name in enumerate( file_list ):
        if 'cat' in file_name:
            if file_index % 6 == 0:
                fVal.write( file_name + ' 0\n' )
            else:
                fTrain.write( file_name + ' 0\n' )
        if 'dog' in file_name:
            if file_index % 6 == 0:
                fVal.write( file_name + ' 1\n' )
            else:
                fTrain.write( file_name + ' 1\n' )

    fVal.close()
    fTrain.close()

# ---- Define 'main' function as the entry point for this script -------------

if __name__ == '__main__':
    main()

# ==== End of file ===========================================================
