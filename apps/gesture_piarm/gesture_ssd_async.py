#!/usr/bin/env python
"""
 Copyright (C) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

 NOTICES: notices are commented on the line of, signifying chagnes to the original Apache 2.0 code provided by Intel Corporation
"""

from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import time
import logging as log
from openvino.inference_engine import IENetwork, IEPlugin
from mearm_control import MeArmController


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=True, type=str)
    args.add_argument("-i", "--input",
                      help="Required. Path to video file or image. 'cam' for capturing video stream from camera",
                      required=True, type=str)
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with the "
                           "kernels implementations.", type=str, default=None)
    args.add_argument("-pp", "--plugin_dir", help="Optional. Path to a plugin folder", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is "
                           "acceptable. The demo will look for a suitable plugin for device specified. "
                           "Default value is CPU", default="CPU", type=str)
    args.add_argument("--labels", help="Optional. Path to labels mapping file", default=None, type=str)
    args.add_argument("-pt", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
                      default=0.5, type=float)

    return parser


# NOTICE: Funtion for robot control
def robot_control(proposals, big_w, big_h, mearm):
    fives = 0
    sixes = 0

    for proposal in proposals:
        xmin = proposal['pt1'][0]
        ymin = proposal['pt1'][1]
        xmax = proposal['pt2'][0]
        ymax = proposal['pt2'][1]

        if xmin <= 50 and ymin > 50 and ymax < big_h - 50:
            mearm.move('base', 3) # check left
            print('left')
        elif xmin >= 50 and ymin <= 50:
            mearm.move('upper', 1) # check up
            print('up')
        elif xmax >= big_w - 50 and ymin > 50 and ymax < big_h - 50:
            mearm.move('base', 4) # check right
            print('right')
        elif xmin >= 50 and ymax >= big_h - 50:
            mearm.move('upper', 2) # check down
            print('down')
        elif proposal['pred'] == 5:
            fives = fives + 1
        elif proposal['pred'] == 6:
            sixes = sixes + 1

    if fives > 0:
        if fives == 1:
            mearm.move('grip', 5) # check open
            print('open')
        else:
            mearm.move('lower', 7)
            print('out')
    elif sixes > 0:
        if sixes == 1:
            mearm.move('grip', 6) # check close
            print('close')
        else:
            mearm.move('lower', 8)
            print('in')


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    # Plugin initialization for specified device and load extensions library if specified
    log.info("Initializing plugin for {} device...".format(args.device))
    plugin = IEPlugin(device=args.device, plugin_dirs=args.plugin_dir)
    if args.cpu_extension and 'CPU' in args.device:
        plugin.add_cpu_extension(args.cpu_extension)
    # Read IR
    log.info("Reading IR...")
    net = IENetwork(model=model_xml, weights=model_bin)

    if plugin.device == "CPU":
        supported_layers = plugin.get_supported_layers(net)
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(plugin.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in demo's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)
    assert len(net.inputs.keys()) == 1, "Demo supports only single input topologies"
    assert len(net.outputs) == 1, "Demo supports only single output topologies"
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    log.info("Loading IR to the plugin...")
    exec_net = plugin.load(network=net, num_requests=2)
    # Read and pre-process input image
    n, c, h, w = net.inputs[input_blob].shape

    print("Net input shape: " + str((n, c, h, w)))
    print("Net output shape: " + str(net.outputs[out_blob].shape))

    del net
    if args.input == 'cam':
        input_stream = 0
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"
    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.strip() for x in f]
    else:
        labels_map = None

    cap = cv2.VideoCapture(input_stream)

    cur_request_id = 0
    next_request_id = 1

    log.info("Starting inference in async mode...") # NOTICE: Deleted unneeded logs
    is_async_mode = True

    # NOTICE: Mearm Controller Init
    mearm = MeArmController()
    ret, frame = cap.read()

    print("To close the application, press 'q' or 'esc'.")
    while cap.isOpened():
        if is_async_mode:
            ret, next_frame = cap.read()
        else:
            ret, frame = cap.read()
        if not ret:
            break

        # NOTICE: Provide integer values for camera view space
        big_w = int(cap.get(3))
        big_h = int(cap.get(4))

        initial_w = cap.get(3)
        initial_h = cap.get(4)

        # Main sync point:
        # in the truly Async mode we start the NEXT infer request, while waiting for the CURRENT to complete
        # in the regular mode we start the CURRENT request and immediately wait for it's completion
        inf_start = time.time()
        if is_async_mode:
            in_frame = cv2.resize(next_frame, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            exec_net.start_async(request_id=next_request_id, inputs={input_blob: in_frame})
        else:
            in_frame = cv2.resize(frame, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            exec_net.start_async(request_id=cur_request_id, inputs={input_blob: in_frame})
        if exec_net.requests[cur_request_id].wait(-1) == 0:
            inf_end = time.time()
            det_time = inf_end - inf_start

            # NOTICE: Parse detection results of the current request, only need one so choose max prob
            proposals = []
            res = exec_net.requests[cur_request_id].outputs[out_blob]
            for obj in res[0][0]:
                # Draw only objects when probability more than specified threshold
                if obj[2] > args.prob_threshold:
                    best_proposal = obj
                    xmin = int(best_proposal[3] * initial_w)
                    ymin = int(best_proposal[4] * initial_h)
                    xmax = int(best_proposal[5] * initial_w)
                    ymax = int(best_proposal[6] * initial_h)
                    class_id = int(best_proposal[1])

                    # Draw box and label\class_id
                    color = (min(class_id * 12.5, 255), min(class_id * 7, 255), min(class_id * 5, 255))

                    # NOTICE: Draw box based on center of original bounding box
                    xmid = int(abs(xmin + xmax) / 2)
                    ymid = int(abs(ymin + ymax) / 2)
                    cv2.circle(frame, (xmid, ymid), 4, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                    cv2.rectangle(frame, (xmid - 100, ymid - 100), (xmid + 99, ymid + 99), color, 2)

                    det_label = labels_map[class_id] if labels_map else str(class_id)
                    cv2.putText(frame, det_label, (xmid, ymid - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
                    proposals.append({'pred': class_id, 'pt1': (xmid - 100, ymid - 100), 'pt2': (xmid + 99, ymid + 99)})

            # NOTICE: Robot control
            if len(proposals) > 0:
                robot_control(proposals, big_w, big_h, mearm)

            # Regions for commands
            cv2.rectangle(frame, (0, 50), (50, big_h - 50), (255, 255, 255), 2)
            cv2.rectangle(frame, (50, 0), (big_w - 50, 50), (0, 0, 255), 2)
            cv2.rectangle(frame, (50, big_h - 50), (big_w -50, big_h - 1), (255, 0, 0), 2)
            cv2.rectangle(frame, (big_w - 50, 50), (big_w - 1, big_h - 50), (0, 255, 0), 2)

            # NOTICE: Deletion of performance statsDraw performance stats

        # NOTICE: Deletion of time stats
        cv2.imshow("Detection Results", frame)

        if is_async_mode:
            cur_request_id, next_request_id = next_request_id, cur_request_id
            frame = next_frame

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'): # NOTICE: Exit key is q and esc
            break
        if key == 27:
            break
        # NOTICE: Deletion of option to do inference in synchronous mode

    cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(main() or 0)
