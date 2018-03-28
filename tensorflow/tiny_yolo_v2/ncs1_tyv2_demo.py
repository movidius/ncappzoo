from mvnc import mvncapi as mvnc
import math
import cv2
import numpy as np
import tensorflow as tf
from numpy import array
import time
import sys

show= True 
ncsrun= True
tfrun= False
timestamps=[]

def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(('tiny-yolo-voc.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image):
  image = array(image).reshape(1, 416, 416, 3)
  """Runs inference on an image.
  Args:
    image: Image file name.
  Returns:
    Nothing
  """
  #if not tf.gfile.Exists(image):
  #  tf.logging.fatal('File does not exist %s', image)
  #image_data = tf.gfile.FastGFile(image, 'rb').read()


  with tf.Session() as sess:
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")

      # Some useful tensors:
    feed_dict = {images_placeholder: image}
    output_tensor = sess.graph.get_tensor_by_name('output:0')
    #predictions = sess.run(output_tensor,
    #                       {'input:0': image_data})
    predictions = sess.run(output_tensor,feed_dict=feed_dict)
    return predictions

def sigmoid(p):
    return 1.0 / (1 + math.exp(-p * 1.0))


def overlap(x1, w1, x2, w2):  # x1 ,x2 are two box center x
    left = max(x1 - w1 / 2.0, x2 - w2 / 2.0)
    right = min(x1 + w1 / 2.0, x2 + w2 / 2.0)
    return right - left


def cal_iou(box, truth):
    w = overlap(box[0], box[2], truth[0], truth[2])
    h = overlap(box[1], box[3], truth[1], truth[3])
    if w < 0 or h < 0:
        return 0
    inter_area = w * h
    union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area
    return inter_area * 1.0 / union_area


def apply_nms(boxes, thres):
    sorted_boxes = sorted(boxes, key=lambda d: d[7])[::-1]
    p = dict()
    for i in range(len(sorted_boxes)):
        if i in p:
            continue

        truth = sorted_boxes[i]
        for j in range(i + 1, len(sorted_boxes)):
            if j in p:
                continue
            box = sorted_boxes[j]
            iou = cal_iou(box, truth)
            #print ("iou: ", iou)
            if iou >= thres:
                p[j] = 1

    res = list()

    for i in range(len(sorted_boxes)):
        if i not in p:
            res.append(sorted_boxes[i])
    return res

image_num = 1;
def det(image, image_id):
    #image='/data/VOCdevkit/VOC2007/JPEGImages/002805.jpg'
    global image_num    
    img = cv2.imread(image)
    img_cv = img
    img = np.divide(img, 255.0) 
    img = cv2.resize(img, (416, 416), cv2.INTER_LINEAR)
    img = img[:,:,::-1]
    #imgcp = img
    img = img.astype(np.float32)
    if ncsrun:
        start = time.time()
        #graph.queue_inference_with_fifo_elem(fifoIn, fifoOut, img.astype(np.float16), descIn, 'user object')
        ##########################################################
        graph.LoadTensor(img.astype(np.float16), 'user object')
        output, userobj = graph.GetResult()
        end = time.time()
        print("time taken for inference", end - start)
        timestamps.append(end - start)
    elif tfrun:
        output=run_inference_on_image(img)

    #print('the results are ', output.shape)
    res = output.astype(np.float32)
    res=np.reshape(res,(13,13,125))


    #print("this is the res.shape", res[:,:,0],res.shape )
    #quit()
    swap = np.zeros((13 * 13, 5, 25))
    #print("this is the shape of swap", swap.shape)
    # change
    index = 0
    for h in range(13):
        for w in range(13):
            for c in range(125):
                #print("This is res value", res[c][h][w])
                i=h*13 + w
                j = int(c/25)
                k = c%25
                #print("these are the values of i j k", i, j, k)
                #swap[h * 13 + w][c / 25][c % 25] = res[c][h][w]
                swap[i][j][k]=res[h][w][c]

    biases = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]

    boxes = list()
    for h in range(13):
        for w in range(13):
            for n in range(5):
                box = list();
                cls = list();
                s = 0;
                x = (w + sigmoid(swap[h * 13 + w][n][0])) / 13.0;
                y = (h + sigmoid(swap[h * 13 + w][n][1])) / 13.0;
                ww = (math.exp(swap[h * 13 + w][n][2]) * biases[2 * n]) / 13.0;
                hh = (math.exp(swap[h * 13 + w][n][3]) * biases[2 * n + 1]) / 13.0;
                obj_score = sigmoid(swap[h * 13 + w][n][4]);
                for p in range(20):
                    cls.append(swap[h * 13 + w][n][5 + p]);

                large = max(cls);
                for i in range(len(cls)):
                    cls[i] = math.exp(cls[i] - large);

                s = sum(cls);
                for i in range(len(cls)):
                    cls[i] = cls[i] * 1.0 / s;

                box.append(x);
                box.append(y);
                box.append(ww);
                box.append(hh);
                box.append(cls.index(max(cls)) + 1)
                box.append(obj_score);
                box.append(max(cls));
                box.append(obj_score * max(cls))
          #      print("these are the values of box 5 6", box[5], box[6])
                if box[5] * box[6] > 0.1:
                    boxes.append(box);
    res = apply_nms(boxes, 0.35)
    label_name = {0: "bg", 1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle", 6: "bus", 7: "car",
                  8: "cat", 9: "chair", 10: "cow", 11: "diningtable", 12: "dog", 13: "horse", 14: "motorbike",
                  15: "person", 16: "pottedplant", 17: "sheep", 18: "sofa", 19: "train", 20: "tvmonitor"}
    w = img_cv.shape[1]
    h = img_cv.shape[0]

    
#    res_name = "./tyv2results/comp4_det_test_";
    for box in res:
        #print("this is the type and value", type(box[4]), box[4])
#        name = res_name + label_name[box[4]]
        # print name
#        fid = open(name + ".txt", 'a')
#        fid.write(image_id[:-4])
#        fid.write(' ')
#        fid.write(str(box[5] * box[6]))
#        fid.write(' ')
        xmin = (box[0] - box[2] / 2.0) * w;
        xmax = (box[0] + box[2] / 2.0) * w;
        ymin = (box[1] - box[3] / 2.0) * h;
        ymax = (box[1] + box[3] / 2.0) * h;
        if xmin < 0:
            xmin = 0
        if xmax > w:
            xmax = w
        if ymin < 0:
            ymin = 0
        if ymax > h:
            ymax = h

#        fid.write(str(xmin))
#        fid.write(' ')
#        fid.write(str(ymin))
#        fid.write(' ')
#        fid.write(str(xmax))
#        fid.write(' ')
#        fid.write(str(ymax))
#        fid.write('\n')
#        fid.close()
        if show:
            cv2.rectangle(img_cv,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,255,0),2)
            #print (label_name[box[4]],xmin, ymin, xmax, ymax)
            label_size = cv2.getTextSize(label_name[box[4]] + " " + str("{0:.2f}".format(box[5]*box[6])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,1)[0]
            label_left = int(xmin)
            label_top = int(ymin) - label_size[1]
            label_right = label_left + label_size[0]
            label_bottom = label_top + label_size[1]
            cv2.rectangle(img_cv, (label_left, label_top-5), (label_right+1, label_bottom-2), (70, 120, 70), -1) 
            cv2.putText(img_cv, label_name[box[4]] + " " + str("{0:.2f}".format(box[5]*box[6])), (int(xmin), int(ymin-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            #cv2.putText(img_cv, res[i][0] + ' : %.2f' % res[i][5],(xmin+5,ymin-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
    if show:
        cv2.imshow('YOLO detection',img_cv)
        cv2.imwrite("yolov2_" + str(image_num) + ".jpg", img_cv)
        image_num = image_num + 1
        cv2.waitKey(10000)


if ncsrun:
#images_path='/data/VOC/VOCtest2007/VOCdevkit/VOC2007/JPEGImages/'

# ***************************************************************
# configure the NCS
# ***************************************************************
   mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 2)

# ***************************************************************
# Get a list of ALL the sticks that are plugged in
# ***************************************************************
   devices = mvnc.EnumerateDevices()
   if len(devices) == 0:
        print('No devices found')
        quit()

# ***************************************************************
# Pick the first stick to run the network
# ***************************************************************
   device = mvnc.Device(devices[0])

# ***************************************************************
# Open the NCS
# ***************************************************************
   device.OpenDevice()
#print("*******************device opened")
#time.sleep(10)
#print("device open sleep done***************")
   network_blob=sys.argv[1]

#Load blob
   with open(network_blob, mode='rb') as f:
        graphfile = f.read()

   graph = device.AllocateGraph(graphfile);
   #device.graph_allocate(graph, graphfile)

# Initialize Fifos
   #fifoIn = mvnc.Fifo(mvnc.FifoType.HOST_RW)
   #fifoOut = mvnc.Fifo(mvnc.FifoType.HOST_RW)

# Get tensor descriptors & desc count
   ##descIn = graph.get_option(mvnc.GraphOptionClass0.INPUT_TENSOR_DESCRIPTORS)
   #descOut = graph.get_option(mvnc.GraphOptionClass0.OUTPUT_TENSOR_DESCRIPTORS)
   #descInCnt = graph.get_option(mvnc.GraphOptionClass0.INPUT_COUNT)
   #descOutCnt = graph.get_option(mvnc.GraphOptionClass0.OUTPUT_COUNT)

# Create Fifos
   #fifoIn.create(device, descIn, 2)
   #fifoOut.create(device, descOut, 4)
if tfrun:
    create_graph()

count=0

index = 0;
for line in open('test.txt', 'r'):
    index += 1
    if index>8:
        break
    if index%100==0:
        print("this is index", index)
    image_name = line.split(' ')[0]
    image_id = image_name.split('/')[-1]
    det(image_name, image_id)

if ncsrun:
   graph.DeallocateGraph()
   device.CloseDevice()
print('Finished')

# for FPS
total = 0
for i in range(len(timestamps)):
    t = timestamps[i]
#    print("run {0} time {1:0.3f} sec.".format(i, t))
    total = total + t
print("total time for {0} frames = {1:0.3f} sec., {2:0.1f} FPS".format(len(timestamps), total, len(timestamps) / total))
print("average inference time: ", total/len(timestamps))
print('Finished FPS')
