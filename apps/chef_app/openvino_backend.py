from flask import Flask, request, jsonify, url_for
from openvino.inference_engine import IEPlugin, IENetwork
from celery import Celery
import cv2
import numpy as np
import base64
from PIL import Image
import io


app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# CORS Agreement Code Snippet
@app.after_request
def after_request(response):
    # Make sure to change localhost:3000 to the actual host:port you want
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

@celery.task(bind=True)
def object_detection(self, uri, model, dev):
    # Pre processing base64 encoded picture
    encoded_data = uri.split(',')[1]
    decoded_data = base64.b64decode(encoded_data)
    pil_image = Image.open(io.BytesIO(decoded_data))
    obj_frame = cv2.cvtColor(np.array(pil_image), cv2.IMREAD_COLOR)

    # Getting Model
    path_to_xml = None
    path_to_bin = None
    if model.lower() == 'food':
        model = 'ssd_inception_v2_mo'
        if dev.lower() == 'cpu':
            path_to_xml = './' + str(model.lower()) + '_fp32/frozen_inference_graph.xml'
            path_to_bin = './' + str(model.lower()) + '_fp32/frozen_inference_graph.bin'
        elif dev.lower() == 'myriad':
            path_to_xml = './' + str(model.lower()) + '_fp16/frozen_inference_graph.xml'
            path_to_bin = './' + str(model.lower()) + '_fp16/frozen_inference_graph.bin'
        else:
            return {'result': '', 'info_state': str(dev) + ' not a valid device', 'status': 'incomplete'}
    else:
        return {'result': '', 'info_state': str(model) + 'not a valid model', 'status': 'incomplete'}

    # Object Detection via OpenVINO
    net = IENetwork(model=path_to_xml, weights=path_to_bin)
    input_layer = next(iter(net.inputs))
    output_layer = next(iter(net.outputs))
    n, c, h, w = net.inputs[input_layer].shape
    obj_in_frame = cv2.resize(obj_frame, (w, h))
    obj_in_frame = obj_in_frame.transpose((2, 0, 1))
    obj_in_frame = obj_in_frame.reshape((n, c, h, w))

    obj_plugin = IEPlugin(device=dev.upper())
    if dev.lower() == 'cpu':
        ext = '/opt/intel/openvino/deployment_tools/inference_engine/'
        ext = ext + 'lib/intel64/libcpu_extension_avx2.so'
        obj_plugin.add_cpu_extension(ext)
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
            cv2.rectangle(obj_frame, (xmin, ymin), (xmax, ymax), color, 2)

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
    task = object_detection.apply_async(args=[img_base64, model, device])
    return jsonify({'task_id': task.id}), 202

@app.route('/status/<task_id>', methods=['GET'])
def taskstatus(task_id):
    response = {}
    task = object_detection.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {'state': task.state, 'status': 'Pending...'}
    elif task.state != 'FAILURE':
        response = {'state': task.state, 'status': task.info.get('status', '')}
        if 'result' in task.info:
            response['result'] = task.info['result']
        if 'info_state' in task.info:
            response['info_state'] = task.info['info_state']
    else:
        response = {'state': task.state, 'status': str(task.info)}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
