
NCCOMPILE = mo.py

GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
RED = '\033[1;31m'
NOCOLOR = '\033[0m'

NETWORK_NAME = tiny_yolo_v2
MODEL_BASE_NAME = yolov2-tiny-voc
DARKFLOW_FOLDER = darkflow
BUILD_FOLDER = built_graph
CFG_FILENAME = ${MODEL_BASE_NAME}.cfg
WEIGHTS_FILENAME = ${MODEL_BASE_NAME}.weights
TF_MODEL_FILENAME = ${MODEL_BASE_NAME}.pb
PASCAL_VOC_LABELS = pascal_voc_labels.txt
INPUT=../../data/images/nps_chair.png

CLONE_DARKFLOW = git clone https://github.com/thtrieu/darkflow.git
GET_CFG = wget --no-cache -P ./darkflow/cfg/ https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/${CFG_FILENAME}
GET_WEIGHTS = wget --no-cache -P ./darkflow/bin/ https://pjreddie.com/media/files/${WEIGHTS_FILENAME}


.PHONY: all
all: data deps compile_model


.PHONY: deps
deps: clone_darkflow_repo
	@echo $(YELLOW)'\n'${NETWORK_NAME}": Making dependencies..."$(NOCOLOR)
	@if python3 -c 'import pkgutil; exit(not pkgutil.find_loader("cython"))' ; \
	then \
		echo 'Cython found.' ; \
	else \
		echo $(YELLOW)"Cython not found. Please install requirements by entering 'make install-reqs'."$(NOCOLOR) ; \
		exit 1 ; \
	fi
	@if python3 -c 'import pkgutil; exit(not pkgutil.find_loader("darkflow"))' ; \
	then \
		echo 'Darkflow found.' ; \
	else \
		echo $(YELLOW)"Darkflow not found. Please install requirements by entering 'make install-reqs'."$(NOCOLOR) ; \
		exit 1 ; \
	fi; \
	

.PHONY: data
data:
	@echo $(YELLOW)'\n'${NETWORK_NAME}": Downloading required data... No data needed."$(NOCOLOR)


.PHONY: generate_model
generate_model: clone_darkflow_repo download_cfg download_weights
	@echo $(YELLOW)'\n'${NETWORK_NAME}": Generating model..."${NOCOLOR}
	@echo "Checking if Tiny Yolo V2 pb file exists..."
	@if [ -e ${TF_MODEL_FILENAME} ] ; \
	then \
		echo "Tiny Yolo V2 Tensorflow pb file already exists." ; \
	else \
		echo "Tiny Yolo V2 Tensorflow pb file does not exist. Saving model to built_graph folder.\n" ; \
		sed '1d' labels.txt > ${PASCAL_VOC_LABELS} ; \
		flow --model darkflow/cfg/${CFG_FILENAME} --load darkflow/bin/${WEIGHTS_FILENAME} --savepb --labels pascal_voc_labels.txt ; \
		cp ${BUILD_FOLDER}/${TF_MODEL_FILENAME} . ; \
		echo "Tiny Yolo V2 Tensorflow model generation complete." ;\
	fi


.PHONY: clone_darkflow_repo
clone_darkflow_repo:
	@echo $(YELLOW)'\n'${NETWORK_NAME}": Checking if darkflow repo is cloned..."${NOCOLOR}
	@if [ -e ${DARKFLOW_FOLDER} ] ; \
	then \
		echo "darkflow folder already exists.\n" ; \
	else \
		echo "darkflow folder does not exist.\n" ; \
		${CLONE_DARKFLOW} ; \
	fi 


.PHONY: download_cfg
download_cfg:
	@echo $(YELLOW)'\n'${NETWORK_NAME}": Downloading Tiny Yolo v2 cfg file..."${NOCOLOR}
	@if [ -e darkflow/cfg/${CFG_FILENAME} ] ; \
	then \
		echo "cfg file already exist.\n" ; \
	else \
		echo "cfg file does not exists.\n" ; \
		${GET_CFG} ; \
	fi 


.PHONY: download_weights
download_weights:
	@echo $(YELLOW)'\n'${NETWORK_NAME}": Downloading Tiny Yolo v2 weights..."${NOCOLOR}
	@if [ -e darkflow/bin/${WEIGHTS_FILENAME} ] ; \
	then \
		echo "weights file already exists.\n" ; \
	else \
		echo "weights file does not exist.\n" ; \
		${GET_WEIGHTS} ; \
	fi 


.PHONY: compile_model
compile_model: generate_model
	@echo $(YELLOW)'\n'${NETWORK_NAME}": Compiling Model to IR..."${NOCOLOR}
	@echo "Checking OpenVINO environment..."
	@if [ -z "$(INTEL_OPENVINO_DIR)" ] ; \
	then \
		echo "Please initiate the Intel OpenVINO environment by going to the installation directory for openvino and running the setupvars.sh file in the bin folder." ; \
		exit 1 ; \
	else \
		echo "Intel OpenVINO environment is already set!" ; \
	fi
	@if [ -e ${MODEL_BASE_NAME}.xml ] && [ -e ${MODEL_BASE_NAME}.bin ]; \
	then \
		echo " - IR files already exist. Skipping compile step."; \
	else \
		${NCCOMPILE} --input_model=yolov2-tiny-voc.pb --batch=1 --tensorflow_use_custom_operations_config=${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/extensions/front/tf/yolo_v2_tiny_voc.json --reverse_input_channels --scale_value=[255] --data_type=FP16; \
	fi

	

.PHONY: run_py
run_py: data deps compile_model
	@echo $(YELLOW)'\n'${NETWORK_NAME}": Running Python sample..."${NOCOLOR}
	python3 ./${NETWORK_NAME}.py --input=${INPUT}


.PHONY: run
run: run_py



.PHONY: install-reqs
install-reqs: clone_darkflow_repo
	@echo $(YELLOW)'\n'${NETWORK_NAME}": Checking application requirements...\n"$(NOCOLOR)
	@echo "If you would like to uninstall the required packages, please enter 'make uninstall-reqs'"
	@echo "Installing Cython..."
	@pip3 install cython --user ; \
	echo "Installing Darkflow..."
	(cd darkflow; pip3 install . --user)
	

.PHONY: uninstall-reqs
uninstall-reqs: 
	@echo $(YELLOW)'\n'${NETWORK_NAME}": Uninstalling requirements..."$(NOCOLOR)
	@echo "Uninstalling darkflow..."; \
	pip3 uninstall darkflow; \
	@echo "Uninstalling Cython..."; \
	pip3 uninstall cython;


.PHONY: help
help:    
	@echo "\nPossible make targets: ";
	@echo $(YELLOW)"  make run or run_py"$(NOCOLOR)"- Runs Python example.";
	@echo $(YELLOW)"  make help "$(NOCOLOR)"- Shows this message.";
	@echo $(YELLOW)"  make all "$(NOCOLOR)"- Makes everything needed to run, but doesn't run.";
	@echo $(YELLOW)"  make compile_model "$(NOCOLOR)"- Uses the Model Optimizer to compile the model files to IR.";
	@echo $(YELLOW)"  make data "$(NOCOLOR)"- Downloads required data.";
	@echo $(YELLOW)"  make deps "$(NOCOLOR)"- Makes dependencies for project, prepares model etc.";
	@echo $(YELLOW)"  make install-reqs "$(NOCOLOR)"- Installs requirements needed to run this sample on your system.";
	@echo $(YELLOW)"  make uninstall-reqs "$(NOCOLOR)"- Uninstalls requirements that were installed by the sample program.";
	@echo $(YELLOW)"  make clean "$(NOCOLOR)"- Removes all created content.\n"
	@echo " ----- "$(YELLOW)"Model information"$(NOCOLOR)" -----"
	@echo $(YELLOW)" Description: "$(NOCOLOR)
	@echo "  This network does object detection on 20 different classes.";
	@echo $(YELLOW)" Model inputs:"$(NOCOLOR)
	@echo "  name: 'input', shape: [1x3x416x416], Expected color order is BGR after optimization.";
	@echo $(YELLOW)" Model outputs:"$(NOCOLOR)
	@echo "  name: 'output/YoloRegion', shape: [1, 21125]. See Readme for more details on the output.\n";


clean: 
	@echo $(YELLOW)'\n'${APP_NAME}": Cleaning up files..."${NOCOLOR}
	rm -rf darkflow
	rm -rf built_graph
	rm -rf bin
	rm -rf ckpt
	rm -rf sample_img
	rm -f ${TF_MODEL_FILENAME}
	rm -f ${MODEL_BASE_NAME}.xml
	rm -f ${MODEL_BASE_NAME}.bin
	rm -f ${MODEL_BASE_NAME}.mapping
	rm -f ${PASCAL_VOC_LABELS}
	
	
