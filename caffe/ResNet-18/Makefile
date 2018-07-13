#NCCOMPILE = python /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo.py
NCCOMPILE = mo.py


TRAIN_PROTOTXT_FILENAME= train.prototxt
PROTOTXT_FILENAME = deploy.prototxt
PROTOTXT_PATCH_FILENAME= resnet-18.prototxt.patch
GET_TRAIN_PROTOTXT = wget -P . https://raw.githubusercontent.com/HolmesShuan/ResNet-18-Caffemodel-on-ImageNet/3f035441361357a0f213269a5f63a6176ea28ebe/${TRAIN_PROTOTXT_FILENAME}

CAFFEMODEL_FILENAME = resnet-18.caffemodel
CAFFEMODEL_URL = https://github.com/HolmesShuan/ResNet-18-Caffemodel-on-ImageNet

GET_CAFFEMODEL = wget --no-check-certificate 'https://onedrive.live.com/download?cid=7C725726AF404CFD&resid=7C725726AF404CFD%21110&authkey=%21ANF8hWlK9D8LGrk' -O resnet-18.caffemodel

GRAPH_FILENAME_BASE = resnet-18

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
prototxt:
	@echo "\nmaking prototxt"
	@if [ -e ${PROTOTXT_FILENAME} ] ; \
	then \
		echo "Prototxt file already exists, skipping download."; \
	else \
		if ${GET_TRAIN_PROTOTXT} ; \
		then \
			cp ${TRAIN_PROTOTXT_FILENAME} ${PROTOTXT_FILENAME} ; \
			echo "Training prototxt file downloaded, now patch for deployment" ; \
			patch ${PROTOTXT_FILENAME} ${PROTOTXT_PATCH_FILENAME} ;\
		else \
			echo "Error, protoxt file failed to download."; \
			echo "Please check internet connection and retry."; \
		fi; \
	fi  

.PHONY: caffemodel
caffemodel: 
	@echo "\nmaking caffemodel"
	@if [ -e ${CAFFEMODEL_FILENAME} ] ; \
	then \
		echo "caffemodel file already exists, skipping download."; \
	else \
		if ${GET_CAFFEMODEL}; \
		then \
		    echo "Successfully downloaded caffemodel." ; \
		else \
			echo "\n***************" ; \
			echo "Error downloading caffemodel file: ${CAFFEMODEL_FILENAME}" ; \
			echo "Please check internet connection and retry or download manually at: "; \
			echo "\n${CAFFEMODEL_URL}\n"; \
			echo "***************" ; \
			rm -f ${CAFFEMODEL_FILENAME} ; \
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
	@echo "  make compile_cpp - builds the run_cpp executable example";
	@echo "  make run - builds and runs the C++ example program";
	@echo "  make clean - removes all created content"

clean_caffe_model:
	@echo "\nmaking clean_caffe_model"
	rm -f ${PROTOTXT_FILENAME}
	rm -f ${CAFFEMODEL_FILENAME}
	rm -f ${PROTOTXT_FILENAME}
	rm -f ${TRAIN_PROTOTXT_FILENAME}


clean: clean_caffe_model
	@echo "\nmaking clean"
	rm -f ${GRAPH_FILENAME_BASE}.bin
	rm -f ${GRAPH_FILENAME_BASE}.mapping.json
	rm -f ${GRAPH_FILENAME_BASE}.xml


	
