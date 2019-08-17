from flask import Flask, request, jsonify, render_template, redirect, url_for, Response
from argparse import ArgumentParser, SUPPRESS
import logging as log
import webbrowser, threading
from openvino.inference_engine import IEPlugin, IENetwork
import cv2
import numpy as np
import base64
from PIL import Image
import io
import os


app = Flask(__name__, template_folder='templates')
# app.config['EXPLAIN_TEMPLATE_LOADING'] = True

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument("-e", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with the "
                           "kernels implementations.", type=str, default=None)
    args.add_argument("-i", "--app_id",
                      help="Optional. Required for Edamam API.", type=str, default=None)
    args.add_argument("-k", "--app_key",
                      help="Optional. Required for Edamam API.", type=str, default=None)
    return parser

# CORS Agreement Code Snippet
@app.after_request
def after_request(response):
    # Make sure to change localhost:3000 to the actual host:port you want
    response.headers.add('Access-Control-Allow-Origin', 'http://127.0.0.1:5000')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Front end rendering
@app.route('/', methods=['GET', 'POST'])
def index():
    args = build_argparser().parse_args()
    i = args.app_id
    k = args.app_key

    if request.method == 'GET':
        return render_template('index.html', id=i, key=k)

    return redirect(url_for('index'), key=k, id=i)

# Backend Test Route
@app.route('/test')
def test():
    return jsonify({"message": 'test successful'}), 200

# Backend for detections
def object_detection(uri, model, dev):
    # Getting cpu extension just in case from arg parser
    args = build_argparser().parse_args()
    extension = args.cpu_extension

    # Pre processing base64 encoded picture
    encoded_data = uri.split(',')[1]
    decoded_data = base64.b64decode(encoded_data)
    pil_image = Image.open(io.BytesIO(decoded_data))
    obj_frame = cv2.cvtColor(np.array(pil_image), cv2.IMREAD_COLOR)

    # Getting Model
    path_to_xml = None
    path_to_bin = None
    if model.lower() == 'food':
        if dev.lower() == 'cpu':
            path_to_xml = './' + str(model.lower()) + '_model_fp32/frozen_inference_graph.xml'
            path_to_bin = './' + str(model.lower()) + '_model_fp32/frozen_inference_graph.bin'
        elif dev.lower() == 'myriad':
            path_to_xml = './' + str(model.lower()) + '_model_fp16/frozen_inference_graph.xml'
            path_to_bin = './' + str(model.lower()) + '_model_fp16/frozen_inference_graph.bin'
        else:
            return {'result': '', 'info_state': str(dev) + ' not a valid device', 'status': 'incomplete'}
    else:
        return {'result': '', 'info_state': str(model) + 'not a valid model', 'status': 'incomplete'}

    # Object Detection via OpenVINO
    net = IENetwork(model=path_to_xml, weights=path_to_bin)
    input_layer = next(iter(net.inputs))
    output_layer = next(iter(net.outputs))
    n, c, h, w = net.inputs[input_layer].shape

    # Pre-processing data
    obj_in_frame = cv2.resize(obj_frame, (w, h))
    obj_in_frame = obj_in_frame.transpose((2, 0, 1))
    obj_in_frame = obj_in_frame.reshape((n, c, h, w))

    obj_plugin = IEPlugin(device=dev.upper())
    if dev.lower() == 'cpu':
        obj_plugin.add_cpu_extension(extension)
    # Loading into network plugin
    obj_exec_net = obj_plugin.load(network=net, num_requests=1)
    del net

    obj_res = obj_exec_net.infer({'image_tensor': obj_in_frame})
    obj_det = obj_res[output_layer]
    initial_w = obj_frame.shape[1]
    initial_h = obj_frame.shape[0]

    preds = []
    for obj in obj_det[0][0]:
    # Draw only objects when probability more than specified threshold
        if obj[2] > 0.5:
            xmin = int(obj[3] * initial_w)
            ymin = int(obj[4] * initial_h)
            xmax = int(obj[5] * initial_w)
            ymax = int(obj[6] * initial_h)
            class_id = int(obj[1])
            preds.append(class_id)
            # Draw box and label\class_id
            color = (min(class_id * 12.5, 255), min(class_id * 7, 255), min(class_id * 5, 255))
            cv2.rectangle(obj_frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), thickness=24)

    out_frame = cv2.cvtColor(obj_frame, cv2.COLOR_BGR2RGB)

    # Conversion back to base64
    jpg_header = 'data:image/jpeg;base64,'
    reval, buffer = cv2.imencode('.jpg', out_frame)
    encoded_retval = str(base64.b64encode(buffer))
    encoded_retval = encoded_retval.replace("\\", "")
    encoded_retval = encoded_retval.replace("b'", "")
    encoded_retval = encoded_retval.replace("'", "")

    preds_retval = ""
    if len(preds) > 0:
        preds_retval = str(preds[0])
        for i in range(1, len(preds)):
            preds_retval = preds_retval + "," + str(preds[i])

    return {'result': jpg_header + encoded_retval, 'info_state': preds_retval, 'status': 'completed'}

@app.route('/detect', methods=['POST'])
def detect():
    req_json = request.get_json()
    img_base64 = req_json['img']
    model = req_json['model_type']
    device = req_json['device']
    task = object_detection(img_base64, model, device)
    return jsonify(task), 202

if __name__ == '__main__':
    threading.Timer(2.25, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(debug=False)
