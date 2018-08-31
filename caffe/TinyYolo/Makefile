#NCCOMPILE = python /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo.py
NCCOMPILE = mo.py

PROTOTXT_FILENAME= tiny-yolo-v1.prototxt
GET_PROTOTXT = wget --no-cache -P . http://ncs-forum-uploads.s3.amazonaws.com/ncappzoo/tiny_yolo/${PROTOTXT_FILENAME}

CAFFEMODEL_FILENAME = tiny-yolo-v1_53000.caffemodel
GET_CAFFEMODEL = wget --no-cache -P . -N http://ncs-forum-uploads.s3.amazonaws.com/ncappzoo/tiny_yolo/${CAFFEMODEL_FILENAME}


GRAPH_FILENAME_BASE =  tiny-yolo-v1_53000


.PHONY: all
all: deps data compile_model compile_cpp

.PHONY: prototxt
prototxt: 
	@echo "\nmaking prototxt"
	@if [ -e ${PROTOTXT_FILENAME} ] ; \
	then \
		echo "Prototxt file already exists"; \
	else \
		echo "Downloading Prototxt file"; \
		${GET_PROTOTXT}; \
		if [ -e ${PROTOTXT_FILENAME} ] ; \
		then \
			echo "got prototext file." ; \
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
profile: 
	@echo "\nProfile not implemented yet"

.PHONY: compile_model
compile_model: deps
	@echo "\nmaking compile"; \
	if [ -e ${GRAPH_FILENAME_BASE}.xml ] ; \
	then \
		echo "compiled model file already exists, skipping compile."; \
	else \
		${NCCOMPILE} --data_type FP16 --input_model ${CAFFEMODEL_FILENAME} --input_proto ${PROTOTXT_FILENAME};\
	fi

.PHONY: deps
deps: caffemodel prototxt
	@echo "\nmaking deps"

.PHONY: data
data: 
	@echo "\nmaking data - no data required for this project"

.PHONY: check
check: 
	@echo "\ncheck not implemented yet."

.PHONY: compile_cpp
compile_cpp: 
	@echo "\ncompile_cpp not implemented yet."

.PHONY: run
run: deps compile_model compile_cpp
	@echo "\nrun not implemented yet."

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
	@echo "  make compile_cpp - builds the run_cpp executable example";
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

