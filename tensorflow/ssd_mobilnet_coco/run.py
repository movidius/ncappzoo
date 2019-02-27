from mvnc import mvncapi as mvnc
import numpy as np
import cv2
import argparse
import time

ARGS = None

def arg_parse():
    global ARGS
    parser = argparse.ArgumentParser(description='SSD Mobilenet Parser')
    parser.add_argument('--ncsgraph', help='path to the ncsgraph', type=str, nargs='?', default='ssd_mob_v1_graph', const='ssd_mob_v1_graph')
    parser.add_argument('--dim', help='default is 300', type=int, nargs='?',
                        default=300, const=300)
    parser.add_argument('--image', help='default is 512_Cellphone.jpg', type=str, nargs=1, default='../../data/images/512_Cellphone.jpg')
    parser.add_argument('--score', help='default is 0.65', type=float, nargs=1, default=0.65)
    ARGS = parser.parse_args()    


def main():
    arg_parse()

    BLOB_FILE = ARGS.ncsgraph
    INPUT_IMAGE = ARGS.image
    INPUT_SIZE = ARGS.dim
    SCORE_THRESHOLD = ARGS.score

    # Read all of the class labels
    with open('./coco-labels-2014_2017.txt', 'r') as f:
        CLASSES = f.readlines()

    # Configure the NCS log level
    mvnc.global_set_option(mvnc.GlobalOption.RW_LOG_LEVEL, 2)

    # Get a list of ALL the sticks that are plugged in
    devices = mvnc.enumerate_devices()
    if len(devices) == 0:
       print('No devices found')
       quit()

    # Pick the first stick to run the network
    device = mvnc.Device(devices[0])

    # Open the NCS
    device.open()    
    # Create graph object by reading in the Movidius graph file
    with open(BLOB_FILE, mode='rb') as f:
       graphfile = f.read()
    graph = mvnc.Graph(BLOB_FILE)
    # Allocate the graph
    fifoIn, fifoOut = graph.allocate_with_fifos(device, graphfile)

    # Call the function that will run the inference
    detect(CLASSES, graph, fifoIn, fifoOut, INPUT_IMAGE, INPUT_SIZE, SCORE_THRESHOLD)

    # Clean up 
    fifoIn.destroy()
    fifoOut.destroy()
    graph.destroy()
    device.close()
    print('Finished')


def preprocess(src, INPUT_DIM):
    img = cv2.resize(src, (INPUT_DIM, INPUT_DIM))
    img = img - 127.5
    img = img * 0.007843
    return img


def postprocess_ncs(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    size = len(out)
    array_size = int(size / 7)
    boxes = np.reshape(out.astype(np.float32), [array_size, 7])
    num_valid_objs = int(boxes[0][0])
    return (num_valid_objs, boxes) 


def detect(CLASSES, graph, fifoIn, fifoOut, IMG_FILE, INPUT_DIM, SCORE_THRESHOLD):
    orig_img = cv2.imread(IMG_FILE)
    img = preprocess(orig_img, INPUT_DIM)
    img = img.astype(np.float32)
    
    # Send the image to the ncs for processing
    graph.queue_inference_with_fifo_elem(fifoIn, fifoOut, img, 'user object')
    # Retrieve results
    output, userobj = fifoOut.read_elem()

    num_valid_objs, boxes = postprocess_ncs(orig_img, output)

    image_h = orig_img.shape[0]
    image_w = orig_img.shape[1]
    
    # Cycle through each detected object and set up the text and bounding boxes for each
    for obj in range(num_valid_objs): 
        obj_num = obj + 1
        box = boxes[obj_num]
        class_id = box[1]
        score = box[2]
        if score >= SCORE_THRESHOLD:
            if np.isnan(box).any() or np.isinf(box).any():
                continue
            xmin = max(int(box[3] * image_w),0)
            ymin = max(int(box[4] * image_h),0)
            xmax = max(int(box[5] * image_w),0)
            ymax = max(int(box[6] * image_h),0)

            box_top_left = (xmin, ymin)
            box_bot_right = (xmax, ymax)
            
            cv2.rectangle(orig_img, box_top_left, box_bot_right, (0,255,0))
            p3 = (max(box_top_left[0], 15), max(box_top_left[1], 15))
            obj_class = CLASSES[int(class_id)][:len(CLASSES[int(class_id)]) - 1]
            title = obj_class + " " +  str(round(score,2))

            # Greenish background for the display text for visibility
            label_background_color = (70, 120, 70)
            # Get label size and draw rectangle box for text background
            label_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)[0]
            label_left = xmin
            label_top = ymin - label_size[1]
            label_right = label_left + label_size[0]
            label_bottom = label_top + label_size[1]
            cv2.rectangle(orig_img, (label_left, label_top-1),(label_right+1, label_bottom+1), label_background_color, -1)
            # Display the display text on the image
            cv2.putText(orig_img, title, p3, cv2.FONT_ITALIC, 0.8, (0, 255, 0), 1)
            
    cv2.imshow("TensorFlow SSD Mobilenet Coco", orig_img)
    k = cv2.waitKey(0) & 0xff
    # Exit if ESC pressed
    if k == 27 : return False

    
if __name__ == "__main__":
    main()
    
