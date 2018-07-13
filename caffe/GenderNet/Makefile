#NCCOMPILE = python /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo.py
NCCOMPILE = mo.py

PROTOTXT_FILENAME= deploy.prototxt
GET_PROTOTXT =  wget -P . https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/gender_net_definitions/deploy.prototxt

CAFFEMODEL_FILENAME = gender_net.caffemodel
GET_CAFFEMODEL = wget -P . -N https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel

GRAPH_FILENAME_BASE = gender_net

SIMPLE_CLASSIFIER_RELATIVE_DIR = ../../apps/simple_classifier
IMAGES_RELATIVE_DIR = .
TEST_IMAGE_FILENAME = image.jpg

GET_TEST_IMAGE = wget -O image.jpg -N http://vis-www.cs.umass.edu/lfw/images/Talisa_Bratt/Talisa_Bratt_0001.jpg

.PHONY: all
all: deps data compile_model compile_cpp

.PHONY: data
data:
	@echo "\nmaking data"
	test -f ${TEST_IMAGE_FILENAME} || ${GET_TEST_IMAGE}
	@(cd ../../data/age_gender; make)


.PHONY: prototxt
prototxt: 
	test -f ${PROTOTXT_FILENAME} || (${GET_PROTOTXT}; awk 'NR <2 {print}' < ${PROTOTXT_FILENAME} > temp; cat input_shape.prototxt >> temp; awk 'NR > 6 {print}' < ${PROTOTXT_FILENAME} >> temp; mv temp ${PROTOTXT_FILENAME})

.PHONY: caffemodel
caffemodel: 
	test -f ${CAFFEMODEL_FILENAME} || ${GET_CAFFEMODEL}

.PHONY: profile
profile: prototxt
	@echo "\nProfile not implemented yet"

.PHONY: compile_model
compile_model: deps
	@echo "\nmaking compile"
	${NCCOMPILE} --data_type FP16 --input_model ${CAFFEMODEL_FILENAME} --input_proto ${PROTOTXT_FILENAME}

.PHONY: deps
deps: caffemodel prototxt
	@echo "\nmaking deps"

.PHONY: check
check: prototxt caffemodel
	@echo "\nCheck not implemented yet."

.PHONY: run
run: run_cpp 
	@echo "\nmaking run"

.PHONY:	compile_cpp
compile_cpp:  
	@echo "\nmaking compile_cpp"
	(cd ${SIMPLE_CLASSIFIER_RELATIVE_DIR}; make compile_cpp; cd -)

.PHONY: run_cpp
run_cpp: all
	@echo "\nmaking run_cpp"
	${SIMPLE_CLASSIFIER_RELATIVE_DIR}/build/simple_classifier ${GRAPH_FILENAME_BASE}.xml ${IMAGES_RELATIVE_DIR}/${TEST_IMAGE_FILENAME}; 


.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - make all - everything needed to run, but doesn't run";
	@echo "  make data - downloads required data.";
	@echo "  make deps - makes dependencies for project, prepares model etc.";
	@echo "  make prototxt - downloads and adds input shape to Caffe prototxt file";
	@echo "  make caffemodel - downloads the caffemodel for the network"
	@echo "  make compile_model - runs model compiler for the network";
	@echo "  make compile_cpp - builds the C++ example example";
	@echo "  make run - builds and runs the C++ example program";
	@echo "  make clean - removes all created content"

clean_caffe_model:
	@echo "\nmaking clean_caffe_model"
	rm -f ${PROTOTXT_FILENAME}
	rm -f ${CAFFEMODEL_FILENAME}

clean: clean_caffe_model
	@echo "\nmaking clean"
	rm -f ${GRAPH_FILENAME_BASE}.bin
	rm -f ${GRAPH_FILENAME_BASE}.mapping.json
	rm -f ${GRAPH_FILENAME_BASE}.xml
	@rm -f image.jpg

