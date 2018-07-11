
ifneq ($(findstring movidius, $(PYTHONPATH)), movidius)
	export PYTHONPATH:=/opt/movidius/caffe/python:/opt/movidius/mvnc/python:$(PYTHONPATH)
endif

NCCOMPILE = mvNCCompile
NCPROFILE = mvNCProfile
NCCHECK   = mvNCCheck

DARKFLOW_FOLDER = darkflow
BUILD_FOLDER = built_graph
CFG_FILENAME = yolov2-tiny-voc.cfg
WEIGHTS_FILENAME = yolov2-tiny-voc.weights
TF_MODEL_FILENAME = yolov2-tiny-voc.pb
GRAPH_FILENAME = tiny_yolo_v2.graph


CLONE_DARKFLOW = git clone https://github.com/thtrieu/darkflow.git
GET_CFG = wget --no-cache -P ./darkflow/cfg/ https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/${CFG_FILENAME}
GET_WEIGHTS = wget --no-cache -P ./darkflow/bin/ https://pjreddie.com/media/files/${WEIGHTS_FILENAME}

.PHONY: all
all: profile compile 

.PHONY: generate_model
generate_model: clone_darkflow_repo download_cfg download_weights
	@echo "\nGenerating model..."
	@echo "Checking if Tiny Yolo V2 pb file exists..."
	@if [ -e ${TF_MODEL_FILENAME} ] ; \
	then \
		echo "Tiny Yolo V2 Tensorflow pb file already exists." ; \
	else \
		echo "Tiny Yolo V2 Tensorflow pb file does not exist. Saving model to built_graph folder.\n" ; \
		flow --model darkflow/cfg/${CFG_FILENAME} --load darkflow/bin/${WEIGHTS_FILENAME} --savepb --labels labels.txt ; \
		cp ${BUILD_FOLDER}/${TF_MODEL_FILENAME} . ; \
		echo "Tiny Yolo V2 Tensorflow model generation complete." ;\
	fi

.PHONY: clone_darkflow_repo
clone_darkflow_repo:
	@echo "\nChecking darkflow..."
	@if [ -e ${DARKFLOW_FOLDER} ] ; \
	then \
		echo "darkflow folder already exists.\n" ; \
	else \
		echo "darkflow folder does not exist.\n" ; \
		${CLONE_DARKFLOW} ; \
		cd darkflow ; \
		pip3 install . ; \
	fi 

.PHONY: download_cfg
download_cfg:
	@echo "\nChecking Tiny Yolo V2 cfg file..." 
	@if [ -e darkflow/cfg/${CFG_FILENAME} ] ; \
	then \
		echo "cfg file already exist.\n" ; \
	else \
		echo "cfg file does not exists.\n" ; \
		${GET_CFG} ; \
	fi 

.PHONY: download_weights
download_weights:
	@echo "\nChecking Tiny Yolo V2 weights file..." 
	@if [ -e darkflow/bin/${WEIGHTS_FILENAME} ] ; \
	then \
		echo "weights file already exists.\n" ; \
	else \
		echo "weights file does not exist.\n" ; \
		${GET_WEIGHTS} ; \
	fi 


.PHONY: profile
profile: generate_model
	@echo "\nmaking profile"
	${NCPROFILE} ${TF_MODEL_FILENAME} -s 12

.PHONY: browse_profile
browse_profile: profile
	@echo "\nmaking browse_profile"
	@if [ -e output_report.html ] ; \
	then \
		firefox output_report.html & \
	else \
		@echo "***\nError - output_report.html not found" ; \
	fi ; 

.PHONY: compile
compile: generate_model
	@echo "\nmaking compile"
	${NCCOMPILE} -o ${GRAPH_FILENAME} ${TF_MODEL_FILENAME} -s 12

.PHONY: check
check: generate_model
	@echo "\nmaking check"
	${NCCHECK} ${TF_MODEL_FILENAME} -s 12

.PHONY: run_py
run_py: compile
	@echo "\nmaking run_py"
	python3 ./run.py

.PHONY: run
run: compile
	@echo "\nmaking run"
	python3 ./run.py

.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - makes the following: prototxt, profile, compile, check, cpp, run_py, run_cpp";
	@echo "  make prototxt - downloads and adds input shape to Caffe prototxt file";
	@echo "  make caffemodel - downloads the caffemodel for the network"
	@echo "  make compile - runs SDK compiler tool to compile the NCS graph file for the network";
	@echo "  make profile - runs the SDK profiler tool to profile the network creating output_report.html";
	@echo "  make browse_profile - runs the SDK profiler tool and brings up report in browser.";
	@echo "  make run - runs the run.py python example program";
	@echo "  make clean - removes all created content"

clean_caffe_model:
	@echo "\nmaking clean_caffe_model"
	rm -f ${PROTOTXT_FILENAME}
	rm -f ${CAFFEMODEL_FILENAME}

clean: clean_caffe_model
	@echo "\nmaking clean"
	rm -f ${GRAPH_FILENAME}
	rm -f output.gv
	rm -f output.gv.svg
	rm -f output_report.html
	rm -f output_expected.npy
	rm -f zero_weights.caffemodel
	rm -f output_result.npy
	rm -f output_val.csv
	rm -rf darkflow
	rm -rf built_graph
	rm -rf bin
	rm -rf sample_img
	rm -rf ckpt
	rm -f graph
	rm -f ${TF_MODEL_FILENAME}
