#NCCOMPILE = python /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo.py
NCCOMPILE = mo.py

PROTOTXT_FILENAME= deploy.prototxt
GET_PROTOTXT = wget -P . https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt

CAFFEMODEL_FILENAME = bvlc_googlenet.caffemodel
GET_CAFFEMODEL = wget -P . -N http://dl.caffe.berkeleyvision.org/${CAFFEMODEL_FILENAME}

GRAPH_FILENAME_BASE = bvlc_googlenet

SIMPLE_CLASSIFIER_RELATIVE_DIR = ../../apps/simple_classifier
IMAGES_RELATIVE_DIR = ../../data/images
TEST_IMAGE_FILENAME = nps_electric_guitar.png

.PHONY: all
all: deps data compile_model compile_cpp

.PHONY: data
data:
	@echo "\nmaking data"
	(cd ../../data/ilsvrc12; make)
	@sed -i 's/\r//' run.py
	@chmod +x run.py

.PHONY: prototxt
prototxt: data
	@echo "\nmaking prototxt"
	@if [ -e ${PROTOTXT_FILENAME} ] ; \
	then \
		echo "Prototxt file already exists"; \
	else \
		echo "Downloading Prototxt file"; \
		${GET_PROTOTXT}; \
		if [ -e ${PROTOTXT_FILENAME} ] ; \
		then \
			echo "Adding input shape to prototxt file."; \
			awk 'NR <2 {print}' < ${PROTOTXT_FILENAME} > temp; cat input_shape.prototxt >> temp; awk 'NR > 7 {print}' < ${PROTOTXT_FILENAME} >> temp; mv temp ${PROTOTXT_FILENAME}; \
		else \
			echo "***\nError - Could not download prototxt file. Check network and proxy settings \n***\n"; \
			exit 1; \
		fi ; \
	fi  

.PHONY: caffemodel
caffemodel: 
	@echo "\nmaking caffemodel"
	@if [ -e ${CAFFEMODEL_FILENAME} ] ; \
	then \
		echo "caffemodel file already exists"; \
	else \
		echo "Downloading caffemodel file"; \
		${GET_CAFFEMODEL}; \
		if ! [ -e ${CAFFEMODEL_FILENAME} ] ; \
		then \
			echo "***\nError - Could not download caffemodel file. Check network and proxy settings \n***\n"; \
			exit 1; \
		fi ; \
	fi  

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

