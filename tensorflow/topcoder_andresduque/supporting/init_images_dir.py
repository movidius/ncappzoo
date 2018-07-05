'''
Copy the images referenced in CSV file from source folder to a destination folder using a custom folder structure (required for training tensorflow model)
When finish, destination folder will contain multiple subfolders, each folder name will be a label index. Each subfolder will contain all images with the same label or class.

Example output directory of a input directory with only 2 labels:
output_dir/1/ - Will contain all images with label 1 (e.g. dogs)
output_dir/2/ - Will contain all images with label 2 (e.g. cats)
'''

import argparse
import os.path
import csv
from shutil import copyfile

def main(args):
    parser = argparse.ArgumentParser(description='Initialize training images directory')
    parser.add_argument(
      "-source-dir",
      dest="source_dir",
      help="""Folder containing source images"""
    )
    parser.add_argument(
      "-dest-dir",
      dest="dest_dir",
      help="""Destination folder"""
    )
    parser.add_argument(
      "-images-file",
      dest="images_file",
      help="""Required CSV file containing the list of images with ground truth"""
    )
    args = parser.parse_args()
    if not os.path.isdir(args.source_dir):
        print("is not a directory: %s" % args.source_dir)
        return 0
  
    os.system('rm -rf  ' + args.dest_dir)
    if not os.path.exists(args.dest_dir):
        os.makedirs(args.dest_dir)
   
    images = []
    with open(args.images_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        # skip header
        next(reader)
        for row_pos, row in enumerate(reader):
            name = row[0]
            path = os.path.join(args.source_dir, name)
            class_index = row[1] if len(row) > 1 else None
            dir = os.path.join(args.dest_dir, class_index)
            if not os.path.exists(dir):
                os.makedirs(dir)
            copyfile(path, os.path.join(dir, name))
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
