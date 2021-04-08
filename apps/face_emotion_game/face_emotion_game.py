#! /usr/bin/env python3

# Copyright(c) 2019 Intel Corporation.
# License: MIT See LICENSE file in root directory.

from argparse import ArgumentParser, SUPPRESS
from openvino.inference_engine import IENetwork, IEPlugin, IECore
import cv2
import logging as log
import numpy as np
import random
import os
import sys
import time
import pickle

# Specify target device
score = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
totalSecond = 7
minSecond = 1.0
startTime = 0.0
timeElapsed = 0.0
pick_emoji = 0

RED_COLOR = (255, 0, 0)
GREEN_COLOR = (50, 255, 50)
DARK_GREEN_COLOR = (10, 150, 50)
YELLOW_COLOR = (50, 255, 255)


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--mirror", action="store_true", help="Flip camera")
    args.add_argument("-fps", "--show_fps", action="store_true", help="Show fps information on top of camera view")
    return parser


def show_countdown(frame, emotionLabel):
    global totalSecond
    global minSecond
    global startTime
    global timeElapsed
    global pick_emoji

    mid_cam = (int) (FRAME_WIDTH / 2) - 55
    emoji = cv2.imread(os.getcwd() + '/src/images/{}.png'.format(emotionLabel[pick_emoji]))
    smile = cv2.addWeighted(frame[5:75, 5 + mid_cam:55 + mid_cam, :], 0.1, emoji, 1, 0)
    frame[5:75, 5 + mid_cam:55 + mid_cam] = smile
    
    if totalSecond >= minSecond:
        if totalSecond <= 5:
            cv2.putText(frame, str(totalSecond), (mid_cam + 60, 40), cv2.FONT_HERSHEY_DUPLEX, 1, GREEN_COLOR, 2, 1)

        timeElapsed += (time.time() - startTime)
        if timeElapsed >= 1 and (timeElapsed - (time.time() - startTime) >= 1):
            totalSecond -= 1
            timeElapsed = 0
            startTime = time.time()
        if totalSecond == 0:
            return emotionLabel[pick_emoji]
    else:
        totalSecond = 7
        # select random emoji
        pick_emoji = random.randint(0, 4)

    return None


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
        
    face_model_xml = os.getcwd() + "/src/data/face-detection/face-detection-retail-0004.xml"
    face_model_bin = os.path.splitext(face_model_xml)[0] + ".bin"

    emotions_model_xml = os.getcwd() + "/src/data/emotions-recognition/emotions-recognition-retail-0003.xml"
    emotions_model_bin = os.path.splitext(emotions_model_xml)[0] + ".bin"

    global score
    device = "MYRIAD"
    fps = ""
    camera_id = 0
    high_score = 0
    show_user_helper = False

    emotionLabel = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Anger']
    
    cap = cv2.VideoCapture(camera_id)
    log.info("Loading Camera id {}".format(camera_id))

    # Read IR - face detection
    face_net = IENetwork(model=face_model_xml, weights=face_model_bin)
    log.info("Face-Detection network has been loaded:\n\t{}\n\t{}".format(face_model_xml, face_model_bin))

    # Read IR - emotions recognition
    emotion_net = IENetwork(model=emotions_model_xml, weights=emotions_model_bin)
    log.info("Emotions-Recognition network has been loaded:\n\t{}\n\t{}".format(emotions_model_xml, emotions_model_bin))

    log.info("Setting device: {}".format(device))
    plugin = IEPlugin(device=device)

    log.info("Loading Face-Detection model to the plugin")
    face_exec_net = plugin.load(network=face_net)
    # Set configurations for face detection
    face_input_blob = next(iter(face_net.inputs))
    face_out_blob = next(iter(face_net.outputs))    
    
    log.info("Loading Emotions-Recognition model to the plugin")
    emotion_exec_net = plugin.load(network=emotion_net)
    # Set configurations for emotion detection
    emotion_input_blob = next(iter(emotion_net.inputs))
    emotion_out_blob = next(iter(emotion_net.outputs))    
    
    if args.mirror:
        log.info("Using camera mirror")

    try:
        with open('score.dat', 'rb') as file:
            high_score = pickle.load(file)
        file.close()
    except:
        log.warning("Can't read high score!!!")

    log.info("Game is starting...")
    while cap.isOpened():
        t1 = time.time()
        ret_val, img = cap.read()

        if not ret_val:
            break

        if args.mirror:
            img = cv2.flip(img, 1)            
        
        if not show_user_helper:
                
            cv2.putText(img, 'High score: {}'.format(high_score), (FRAME_WIDTH - 200, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, RED_COLOR, 1, 1)        
            cv2.putText(img, 'Your score is: {}'.format(score), (5, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, RED_COLOR, 1, 1)
            picked_emoji = show_countdown(img, emotionLabel)        

            prepimg = cv2.resize(img, (300, 300))
            prepimg = prepimg[np.newaxis, :, :, :]
            prepimg = prepimg.transpose((0, 3, 1, 2))
            face_outputs = face_exec_net.infer(inputs={face_input_blob: prepimg})
            res = face_exec_net.requests[0].outputs[face_out_blob]    

            for detection in res[0][0]:
                confidence = float(detection[2])
                xmin = int(detection[3] * img.shape[1])
                ymin = int(detection[4] * img.shape[0])
                xmax = int(detection[5] * img.shape[1])
                ymax = int(detection[6] * img.shape[0])

                if confidence > 0.7:
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=GREEN_COLOR)

                    if ymin >= 64 and ymax >= 64:
                        emoimg = img[ymin:ymax, xmin:xmax]                    
                        emoimg = cv2.resize(emoimg, (64, 64))                    
                        emoimg = emoimg.transpose((2, 0, 1))
                        emoimg = emoimg.reshape(1, 3, 64, 64)
                        emotion_outputs = emotion_exec_net.infer(inputs={emotion_input_blob: emoimg})
                        res = emotion_exec_net.requests[0].outputs[emotion_out_blob]                                        
                        out_emotion_reshape = res.reshape(-1, 5)
                        emotion_text = emotionLabel[np.argmax(out_emotion_reshape)]

                        cv2.putText(img, emotion_text, (abs(xmin), abs(ymin - 10)), cv2.FONT_HERSHEY_DUPLEX, 0.7, (50, 255, 255), 1, 1)
                      
                        if picked_emoji:
                            log.info("picked: {} recognized: {}".format(emotion_text, picked_emoji))
                        if picked_emoji and picked_emoji == emotion_text:
                            score += 10                        
                        elif picked_emoji and picked_emoji != emotion_text:
                            # save high score                        
                            with open('score.dat', 'wb') as file:
                                high_score = score if score > high_score else high_score
                                pickle.dump(high_score, file)
                                log.info("Your score: {}".format(str(score)))
                            file.close()
                            # reset personal score
                            score = 0
            

            if args.show_fps:
                elapsed_time = time.time() - t1
                fps = "(Playback) {:.1f} FPS".format(1 / elapsed_time)
                cv2.putText(img, fps, (15, FRAME_HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, YELLOW_COLOR, 1, cv2.LINE_AA)
            
            cv2.putText(img, "Hit 'h' for help", (FRAME_WIDTH - 150, FRAME_HEIGHT - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, YELLOW_COLOR, 1, cv2.LINE_AA)
            cv2.putText(img, "Hit 'ESC' or 'q' to Exit", (FRAME_WIDTH - 150, FRAME_HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, YELLOW_COLOR, 1, cv2.LINE_AA)

            cv2.imshow('Face Emotion Game', img)
            
            waitkey = cv2.waitKey(1)
            if waitkey & 0xFF == ord('q') or waitkey == 27:
                # save high score
                with open('score.dat', 'wb') as file:
                    high_score = score if score > high_score else high_score
                    pickle.dump(high_score, file)
                    log.info("High score saved: {}".format(str(high_score)))
                file.close()
                break  # esc or 'q' to quit
            
            if waitkey & 0xFF == ord('h'):
                log.info('Opening help window...')
                show_user_helper = True
                
        # on help window:        
        waitkey = cv2.waitKey(1)
        if waitkey & 0xFF == ord('q') or waitkey == 27:
            log.info('Exiting help window...')
            
            show_user_helper = False
        
        with open ('help.txt', 'r') as helpfile:      
            y0, dy = 50, 15
            for i, line in enumerate(helpfile.read().split("\n")):
                y = y0 + i * dy                
                cv2.putText(img, line, (int(30), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, YELLOW_COLOR, 1, cv2.LINE_AA)
        
        cv2.putText(img, "Hit 'ESC' or 'q' to quit user help", (FRAME_WIDTH - 220, FRAME_HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, YELLOW_COLOR, 1, cv2.LINE_AA)
        cv2.imshow('Face Emotion Game', img)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(main() or 0)
