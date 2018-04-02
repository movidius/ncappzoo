from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import sys
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to checkpoint files')
parser.add_argument('--print_tensor_vals', action='store_true', help='Whether to print all tensor values')
args=parser.parse_args()

print_tensors_in_checkpoint_file(file_name=args.checkpoint_path, tensor_name='', all_tensors=args.print_tensor_vals)