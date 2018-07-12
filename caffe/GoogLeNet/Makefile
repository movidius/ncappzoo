
ifneq ($(findstring movidius, $(PYTHONPATH)), movidius)
	export PYTHONPATH:=/opt/movidius/caffe/python:/opt/movidius/mvnc/python:$(PYTHONPATH)
endif

NCCOMPILE = /opt/intel/computer_vision_sdk_2018.1.265/deployment_tools/model_optimizer/mo.py
#NCPROFILE = mvNCProfile
#NCCHECK   = mvNCCheck


PROTOTXT_FILENAME= deploy.prototxt
GET_PROTOTXT = wget -P . https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt

CAFFEMODEL_FILENAME = bvlc_googlenet.caffemodel
GET_CAFFEMODEL = wget -P . -N http://dl.caffe.berkeleyvision.org/${CAFFEMODEL_FILENAME}

GRAPH_FILENAME = bvlc_googlenet.xml

.PHONY: all
all: deps data compile_cpp 

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
compile_model: prototxt caffemodel
	@echo "\nmaking compile"
	python ${NCCOMPILE} --data_type FP16 --input_model ${CAFFEMODEL_FILENAME} --input_proto ${PROTOTXT_FILENAME}


.PHONY: deps
deps: caffemodel prototxt compile_model
	@echo "\nmaking deps"

.PHONY: check
check: prototxt caffemodel
	@echo "\nCheck not implemented yet."

.PHONY: run
run: run_cpp
	@echo "\nmaking run"

.PHONY: run_py
run_py: compile_model
	@echo "\nmaking run_py"
	./run.py

.PHONY:	compile_cpp
compile_cpp:  
	@echo "\nmaking compile_cpp"
	(mkdir -p build; cd build; cmake ..; make)

.PHONY: run_cpp
run_cpp: deps data compile_cpp
	@echo "\nmaking run_cpp"
	cd build; ./run_googlenet ../${GRAPH_FILENAME} ../../../data/images/nps_electric_guitar.png; cd ..

.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - makes the following: prototxt, profile, compile_model, check, compile_cpp";
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
	rm -f bvlc_googlenet.bin
	rm -f bvlc_googlenet.mapping.json
	rm -f ${GRAPH_FILENAME}
	rm -f cpp/run_cpp
	rm -rf build
