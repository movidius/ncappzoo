
NCCOMPILE = /opt/intel/computer_vision_sdk_2018.1.265/deployment_tools/model_optimizer/mo.py


PROTOTXT_FILENAME= deploy.prototxt
GET_PROTOTXT = wget -P . https://github.com/DeepScale/SqueezeNet/raw/master/SqueezeNet_v1.0/${PROTOTXT_FILENAME}

CAFFEMODEL_FILENAME = squeezenet_v1.0.caffemodel
GET_CAFFEMODEL = wget -P . -N https://github.com/DeepScale/SqueezeNet/raw/master/SqueezeNet_v1.0/${CAFFEMODEL_FILENAME}


GRAPH_FILENAME_BASE = squeezenet_v1.0

.PHONY: all
all: deps data compile_model compile_cpp

.PHONY: data
data:
	@echo "\nmaking data"
	(cd ../../data/ilsvrc12; make)
	@sed -i 's/\r//' run.py
	@chmod +x run.py
	@if [ -e LICENSE.squeezenet ] ; \
	then \
		echo "LICENSE file already exists"; \
	else \
		echo "Downloading LICENSE file"; \
		wget -P . -N https://github.com/DeepScale/SqueezeNet/raw/master/LICENSE; mv LICENSE LICENSE.squeezenet; \
	fi ; \


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
			echo "modifying prototxt file."; \
			sed -i '11s/10/1/' ${PROTOTXT_FILENAME}; sed -i '530s/pad/#pad/' ${PROTOTXT_FILENAME}; \
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
	python ${NCCOMPILE} --data_type FP16 --input_model ${CAFFEMODEL_FILENAME} --input_proto ${PROTOTXT_FILENAME}


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
	(mkdir -p build; cd build; cmake ..; make)

.PHONY: run_cpp
run_cpp: deps data compile_cpp
	@echo "\nmaking run_cpp"
	cd build; ./run_squeezenet ../${GRAPH_FILENAME_BASE}.xml ../../../data/images/nps_electric_guitar.png; cd ..

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
	@echo "  make run_cpp - builds and runs the C++ example program";
	@echo "  make run_py - runs the python example program";
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
	rm -rf build
	rm -f LICENSE.squeezenet

