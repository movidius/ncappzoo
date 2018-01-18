import tensorflow as tf
import inception_resnet_v1 as network
import os
from sys import argv
import sys

image_size=160
central_fraction = 0.875
model_base_file_name = ""

def print_usage():
    print("Usage: ")
    print("python3 convert_facenet.py model_base=<model_base_file_name>")
    print("    where <model_base_file_name> is the base file name of the saved tensorflow model files")
    print("    For example if your model files are: ")
    print("        facenet.index")
    print("        facenet.data-00000-of-00001")
    print("        facenet.meta")
    print("    then <model_base_file_name> is 'facenet' and you would pass model_base=facenet")


# handle the arguments
# return False if program should stop or True if args are ok
def handle_args():
    global model_base_file_name
    model_base_file_name = None
    for an_arg in argv:
        if (an_arg == argv[0]):
            continue

        elif (str(an_arg).lower() == 'help'):
            return False

        elif (str(an_arg).startswith('model_base=')):
            arg, val = str(an_arg).split('=', 1)
            model_base_file_name = str(val)
            print("model base file name is: " + model_base_file_name)
            return True

        else:
            return False

    if (model_base_file_name == None or len(model_base_file_name) < 1):
        return False

    return True



# This function is called from the entry point to do
# all the work.
def main():

    if (not handle_args()):
        # invalid arguments exit program
        print_usage()
        return 1

    with tf.Graph().as_default():
        image = tf.placeholder("float", shape=[1, image_size, image_size, 3], name='input')
        prelogits, _ = network.inference(image, 1.0, phase_train=False)
        normalized = tf.nn.l2_normalize(prelogits, 1, name='l2_normalize')
        output = tf.identity(normalized, name='output')
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            index_file_name = model_base_file_name + ".index"
            data_file_name = model_base_file_name + ".data-00000-of-00001"
            meta_file_name = model_base_file_name + ".meta"
            if (os.path.isfile(index_file_name) and
                os.path.isfile(data_file_name) and
                os.path.isfile(meta_file_name)):
                print('Restoring ' + model_base_file_name)
                saver.restore(sess, model_base_file_name)
            else:
                print('\n')
                print('Error, did not find index, data, and meta files: ')
                print('  ' + index_file_name)
                print('  ' + data_file_name)
                print('  ' + meta_file_name)

                print('These files can be downloaded manually from:')
                print('  https://github.com/davidsandberg/facenet')
                print('    Download: 20170511-185253.zip (web faces), or ')
                print('              20170512-110547.zip (celeb faces)')
                print('after unzipping be sure to rename them to the file names')
                print('above to match the TensorFlow 1.3 file naming.')
                print('Either the celeb faces or web faces will work')
                print('***********************************************************')

            # Save the network for fathom
            saver.save(sess, model_base_file_name + '_ncs/' + model_base_file_name + '_ncs')


# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    sys.exit(main())