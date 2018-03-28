# How to run tiny yolo v2 demo: 
Open a terminal and type in the following command:
```
python3 ncs1_tyv2_demo.py <GRAPH FILE NAME>
``` 

# Original readme

# tinyyolov2
tensorflow tiny yolov2

This is the repo for running tiny yolo v2 on NCS. Steps to create graph file are below:
- Download the tinyolov2 cfg and weight files from darknet repo. 
- Use darkflow to create .pb frozen file for tinyyolov2. 
- Note: Recreating .pb file is a must for each version of NCSDK as NCSDK blob changes with version.
- Note: .pb file is the frozen file and .meta file has the tiny yolo v2 network details such as threshold, classes etc. .meta file is not needed to run the network on ncs.
- Tinyyolov2 weight + cfg file can be downloaded from my googledrive below:
https://drive.google.com/drive/folders/1X6_vjH_tucq1crs5t5KTa9lKgOB5AGWq?usp=sharing

Once .pb is created, compile as below to create a ncs 'graph' object:

mvNCCompile -s 12 tiny-yolo-voc.pb -in=import/input -on=import/output -o tf_tyv2_graph
- Note: NCSDK1.11 needs this import/input import/output
- Note: NCSDK1.12 doesn't need 'import' keyword, just -in=input and -on=output will do

Note that the script ncs_tf_tyv2.py has below options:
- show is set to false. if need to display the results/JPEGs with boxes, set it to True
- Same script can run on NCS or TF. Set ncsrun or tfrun to true/false accordingly. 
- thresholds can be played with. 2 thresholds are used in teh postprocessing. nms threshold (used 0.5) and confidence filter (used 0.35). 
- test.txt has the paths to the annotations and images. (test_2007.txt has full VOC 2007 images/annotations details. If running full VOC 2007 testsing, please update this file as per your local storage)
- for running the script: python3 ncs_tf_tyv2.py (for NCS2, use ncs2_tf_tyv2.py)
- depending on the image used, result files start wtih 'comp4_det_test*' will get created. These files will have the image, confidence score and bounding box details.

