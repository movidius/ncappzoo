#!/usr/bin/python3

# ****************************************************************************
# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.
# ****************************************************************************

# Create labels files for training & validation data 
# Usage: python create-labels.py --data_dir /path/to/data

import os
import sys
import glob
import ntpath
import argparse

ARGS                = None

# ---- Main function (entry point for this script ) --------------------------
def main():

    # Create ground truth labels file for train & validation
    fVal = open( ARGS.data_dir + '/val.txt', 'w' )
    fTrain = open( ARGS.data_dir + '/train.txt', 'w' )

    # Create a list of all files in current directory & sub-directories
    file_list = [ os.path.basename(img) for img in 
                  glob.glob( ARGS.data_dir + '/train/*.jpg' ) ]

    # Create ground truth data, and split dataset to training & validation
    for file_index, file_name in enumerate( file_list ):

        # Assign class ID '0' to 'cats'
        if 'cat' in file_name:
            # Push every 6th cat image to validation dataset
            if file_index % 6 == 0:
                fVal.write( file_name + ' 0\n' )
            else:
                fTrain.write( file_name + ' 0\n' )

        # Assign class ID '1' to 'dogs'
        if 'dog' in file_name:
            # Push every 6th dog image to validation dataset
            if file_index % 6 == 0:
                fVal.write( file_name + ' 1\n' )
            else:
                fTrain.write( file_name + ' 1\n' )

    fVal.close()
    fTrain.close()

# ---- Define 'main' function as the entry point for this script -------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument( '-S', '--data_dir', type=str,
                         default='data',
                         help="Directory where train & test data is downloaded." )

    ARGS = parser.parse_args()

    main()

# ==== End of file ===========================================================
