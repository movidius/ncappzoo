import cv2
from argparse import ArgumentParser, SUPPRESS
import logging as log
import os
import sys
from openvino.inference_engine import IENetwork, IEPlugin

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=True, type=str)
    args.add_argument("-i", "--input",
                      help="Required. Path to video file or image. 'cam' for capturing video stream from camera",
                      required=True, type=str)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is "
                           "acceptable. The demo will look for a suitable plugin for device specified. "
                           "Default value is CPU", default="CPU", type=str)
    args.add_argument("-e", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with the "
                           "kernels implementations.", type=str, default=None)
    args.add_argument("-l", "--labels", help="Optional. Path to labels mapping file", default=None, type=str)

    return parser

def main(path_to_objxml, path_to_objbin, dev, ext, input, labels_map=None):
    # Load network
    obj_net = IENetwork(model=path_to_objxml, weights=path_to_objbin)
    log.info("Loaded network")
    input_layer = next(iter(obj_net.inputs))
    output_layer = next(iter(obj_net.outputs))

    # Pre-process image
    n, c, h, w = obj_net.inputs[input_layer].shape
    obj_frame = cv2.imread(input)
    obj_in_frame = cv2.resize(obj_frame, (w, h))
    obj_in_frame = obj_in_frame.transpose((2, 0, 1))
    obj_in_frame = obj_in_frame.reshape((n, c, h, w))
    log.info("Pre-processed image")

    obj_plugin = IEPlugin(device=dev)
    if dev == 'CPU':
        obj_plugin.add_cpu_extension(ext)
    obj_exec_net = obj_plugin.load(network=obj_net, num_requests=1)
    log.info("Loaded network into plugin")

    # Do inference
    obj_res = obj_exec_net.infer({input_layer: obj_in_frame})
    log.info("Inference successful!")
    obj_det = obj_res[output_layer]

    initial_w = obj_frame.shape[1]
    initial_h = obj_frame.shape[0]
    for obj in obj_det[0][0]:
        # Draw only objects when probability more than specified threshold
        if obj[2] > 0.5:
            xmin = int(obj[3] * initial_w)
            ymin = int(obj[4] * initial_h)
            xmax = int(obj[5] * initial_w)
            ymax = int(obj[6] * initial_h)
            class_id = int(obj[1])

            # Draw box and label\class_id
            color = (min(class_id * 12.5, 255), min(class_id * 7, 255), min(class_id * 5, 255))
            det_label = labels_map[class_id - 1] if labels_map else str(class_id)
            cv2.rectangle(obj_frame, (xmin, ymin), (xmax, ymax), color, 2)
            label_and_prob = det_label + ", " + str(obj[2] * 100) + "%"
            log.info('Detection: ' + label_and_prob)
            #cv2.putText(obj_frame, label_and_prob, (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

    log.info("Hit q to close the window")

    # Resizing to maintainable window
    inHeight = 368
    aspect_ratio = initial_w / initial_h
    inWidth = int(((aspect_ratio*inHeight)*8)//8)
    obj_frame = cv2.resize(obj_frame, (inWidth, inHeight))

    while True:
        cv2.imshow('Detections', obj_frame)
        key = cv2.waitKey(1)
        if (key & 0xFF) == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    device = args.device
    extension = args.cpu_extension
    input = args.input

    labels_map = None
    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.strip() for x in f]
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    log.info("Processed command line arguments")

    main(model_xml, model_bin, device, extension, input, labels_map)
