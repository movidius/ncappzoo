from mvnc import mvncapi as mvnc
import sys
import numpy as np
import cv2
import os

image_filename = './0.jpg'
pnet_graph_filename = './p2838.graph'
onet_graph_filename = './o.graph'

with open(pnet_graph_filename, mode='rb') as rf:
    pgraphfile = rf.read()

with open(onet_graph_filename, mode='rb') as rf:
    ographfile = rf.read()

#mvnc.SetGlobalOption(mvnc.GlobalOption.LOGLEVEL, 2)
devices = mvnc.EnumerateDevices()
if len(devices) < 2:
    print('No enough devices found. Two devices needed for this sample')
    sys.exit(0)

device = mvnc.Device(devices[0])
device.OpenDevice()
device1 = mvnc.Device(devices[1])
device1.OpenDevice()

pgraph = device.AllocateGraph(pgraphfile)
ograph = device1.AllocateGraph(ographfile)
    
def imresample(img, sz):
    im_data = cv2.resize(img, (sz[1], sz[0]), interpolation=cv2.INTER_AREA)
    return im_data

def detect_face(img, threshold = [0.405, 0.8473], factor=0.709):
    # im: input image
    # minsize: minimum of faces' size
    w = 128
    h = 96
    scale_factor = img.shape[1] / w
    img = cv2.resize(img, (w, h))

    total_boxes = np.empty((0, 9))
    im_data = imresample(img, (28, 38))
    im_data = (im_data - 127.5) * 0.0078125
    img_x = np.expand_dims(im_data, 0)
    img_y = np.transpose(img_x, (0, 2, 1, 3))
    pgraph.LoadTensor(img_y[0].astype(np.float16), 'user object')
    out, userobj = pgraph.GetResult()
    out = np.reshape(out, (1, 9, 14, 6))
    boxes, _ = generateBoundingBox(out[0, :, :, 1] - out[0, :, :, 0], out[0, :, :, 2 :].copy(), 0.3, threshold[0])
    # inter-scale nms
    pick = nms(boxes.copy(), 0.5, 'Union')
    if boxes.size > 0 and pick.size > 0:
        boxes = boxes[pick, :]
        total_boxes = np.append(total_boxes, boxes, axis=0)

    total_boxes = np.asarray(total_boxes)
    numbox = total_boxes.shape[0]
    if numbox > 0:
        pick = nms(total_boxes.copy(), 0.7, 'Union')
        total_boxes = total_boxes[pick, :]
        regw = total_boxes[:, 2] - total_boxes[:, 0]
        regh = total_boxes[:, 3] - total_boxes[:, 1]
        qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
        qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
        qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
        qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh
        total_boxes = np.transpose(
            np.vstack([qq1, qq2, qq3, qq4, total_boxes[:, 4]]))
        total_boxes = rerec(total_boxes.copy())
        total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4]).astype(np.int32)
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(
            total_boxes.copy(), w, h)
    
    numbox = total_boxes.shape[0]
    
    if numbox > 0:
        # third stage
        tempimg = np.zeros((48, 48, 3, numbox))
        for k in range(0, numbox):
            tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
            tmp[dy[k] - 1:edy[k], dx[k] - 1:edx[k], :] = img[y[k] - 1:ey[k], x[k] - 1:ex[k], :]
            if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                tempimg[:, :, :, k] = imresample(tmp, (48, 48))

            else:
                return np.empty()
        tempimg = (tempimg - 127.5) * 0.0078125
        tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))

        out = np.zeros((tempimg1.shape[0], 6))

        for k in range(tempimg1.shape[0]):
            ograph.LoadTensor(tempimg1[k].astype(np.float16), 'user object')
            tempout, userobj = ograph.GetResult()
            out[k, :] = tempout[:6]

        out = np.transpose(out)
        score = out[1, :] - out[0, :]
        ipass = np.where(score > threshold[1])
        total_boxes = np.hstack(
            [total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])
        mv = out[2:, ipass[0]]

        w = total_boxes[:, 2] - total_boxes[:, 0] + 1
        h = total_boxes[:, 3] - total_boxes[:, 1] + 1
        if total_boxes.shape[0] > 0:
            total_boxes = bbreg(total_boxes.copy(), np.transpose(mv))
            pick = nms(total_boxes.copy(), 0.7, 'Min')
            total_boxes = total_boxes[pick, :]
            
    rects = [(int(max(0, rect[0]) * scale_factor), int(max(0, rect[1]) * scale_factor),
              int((rect[2] - rect[0]) * scale_factor), int((rect[3] - rect[1]) * scale_factor)) for rect in total_boxes]
    return rects

# function [boundingbox] = bbreg(boundingbox,reg)
def bbreg(boundingbox, reg):
    # calibrate bounding boxes
    if reg.shape[1] == 1:
        reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))

    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1
    b1 = boundingbox[:, 0] + reg[:, 0] * w
    b2 = boundingbox[:, 1] + reg[:, 1] * h
    b3 = boundingbox[:, 2] + reg[:, 2] * w
    b4 = boundingbox[:, 3] + reg[:, 3] * h
    boundingbox[:, 0:4] = np.transpose(np.vstack([b1, b2, b3, b4]))
    return boundingbox


def generateBoundingBox(imap, reg, scale, t):
    # use heatmap to generate bounding boxes
    stride = 2
    cellsize = 12

    imap = np.transpose(imap)
    dx1 = np.transpose(reg[:, :, 0])
    dy1 = np.transpose(reg[:, :, 1])
    dx2 = np.transpose(reg[:, :, 2])
    dy2 = np.transpose(reg[:, :, 3])
    y, x = np.where(imap >= t)
    if y.shape[0] == 1:
        dx1 = np.flipud(dx1)
        dy1 = np.flipud(dy1)
        dx2 = np.flipud(dx2)
        dy2 = np.flipud(dy2)
    score = imap[(y, x)]
    reg = np.transpose(
        np.vstack([dx1[(y, x)], dy1[(y, x)], dx2[(y, x)], dy2[(y, x)]]))
    if reg.size == 0:
        reg = np.empty((0, 3))
    bb = np.transpose(np.vstack([y, x]))
    q1 = np.fix((stride * bb + 1) / scale)
    q2 = np.fix((stride * bb + cellsize - 1 + 1) / scale)
    boundingbox = np.hstack([q1, q2, np.expand_dims(score, 1), reg])
    return boundingbox, reg

# function pick = nms(boxes,threshold,type)
def nms(boxes, threshold, method):
    if boxes.size == 0:
        return np.empty((0, 3))
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    I = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while I.size > 0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]
        xx1 = np.maximum(x1[i], x1[idx])
        yy1 = np.maximum(y1[i], y1[idx])
        xx2 = np.minimum(x2[i], x2[idx])
        yy2 = np.minimum(y2[i], y2[idx])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if method is 'Min':
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        I = I[np.where(o <= threshold)]
    pick = pick[0:counter]
    return pick

# function [dy edy dx edx y ey x ex tmpw tmph] = pad(total_boxes,w,h)
def pad(total_boxes, w, h):
    # compute the padding coordinates (pad the bounding boxes to square)
    tmpw = (total_boxes[:, 2] - total_boxes[:, 0] + 1).astype(np.int32)
    tmph = (total_boxes[:, 3] - total_boxes[:, 1] + 1).astype(np.int32)
    numbox = total_boxes.shape[0]

    dx = np.ones((numbox), dtype=np.int32)
    dy = np.ones((numbox), dtype=np.int32)
    edx = tmpw.copy().astype(np.int32)
    edy = tmph.copy().astype(np.int32)

    x = total_boxes[:, 0].copy().astype(np.int32)
    y = total_boxes[:, 1].copy().astype(np.int32)
    ex = total_boxes[:, 2].copy().astype(np.int32)
    ey = total_boxes[:, 3].copy().astype(np.int32)

    tmp = np.where(ex > w)
    edx.flat[tmp] = np.expand_dims(-ex[tmp] + w + tmpw[tmp], 1)
    ex[tmp] = w

    tmp = np.where(ey > h)
    edy.flat[tmp] = np.expand_dims(-ey[tmp] + h + tmph[tmp], 1)
    ey[tmp] = h

    tmp = np.where(x < 1)
    dx.flat[tmp] = np.expand_dims(2 - x[tmp], 1)
    x[tmp] = 1

    tmp = np.where(y < 1)
    dy.flat[tmp] = np.expand_dims(2 - y[tmp], 1)
    y[tmp] = 1

    return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph

# function [bboxA] = rerec(bboxA)
def rerec(bboxA):
    # convert bboxA to square
    h = bboxA[:, 3] - bboxA[:, 1]
    w = bboxA[:, 2] - bboxA[:, 0]
    l = np.maximum(w, h)
    bboxA[:, 0] = bboxA[:, 0] + w * 0.5 - l * 0.5
    bboxA[:, 1] = bboxA[:, 1] + h * 0.5 - l * 0.5
    bboxA[:, 2:4] = bboxA[:, 0:2] + np.transpose(np.tile(l, (2, 1)))
    return bboxA

def getPaddingSize(img):
    h, w, _ = img.shape
    top, bottom, left, right = (0,0,0,0)

    if w < int(h / 3 * 4):
        tmp = int(h / 3 * 4) - w
        left = tmp // 2
        right = tmp - left
    elif h < int(w / 4 * 3):
        tmp = int(w / 4 * 3) - h
        top = tmp // 2
        bottom = tmp - top
    else:
        pass
    return top, bottom, left, right

image = cv2.imread(image_filename)
top,bottom,left,right = getPaddingSize(image)
image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
rects = detect_face(frame)
if (len(rects) == 0):
    print ("No faces detected.")
elif (len(rects) == 1):
    print ("One face detected.")
else :
    print (len(rects), " faces detected.")
for (i, rect) in enumerate(rects):
    print ("index:", i, " coordinates:", (rect[0], rect[1]), (rect[0] + rect[2], rect[1]+rect[3]))

pgraph.DeallocateGraph()
device.CloseDevice()
ograph.DeallocateGraph()
device1.CloseDevice()
