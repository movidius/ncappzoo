
NCCOMPILE = mo.py

GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
RED = '\033[1;31m'
NOCOLOR = '\033[0m'

DARKNET_MODEL_BASE_NAME = frozen_darknet_yolov3_model

TF_YOLO_REPO_NAME = tensorflow-yolo-v3
CLONE_REPO = git clone https://github.com/mystic123/tensorflow-yolo-v3.git

WEIGHTS_FILE = yolov3-tiny.weights
NETWORK_NAME = tiny_yolo_v3

IR = ${NETWORK_NAME}.xml
LABELS = coco.names
INPUT = ../../data/images/nps_chair.png


.PHONY: all
all: data deps compile_model


.PHONY: deps
deps: convert_weights
	@echo $(YELLOW)'\n'${NETWORK_NAME}": Making dependencies..."$(NOCOLOR)

	
.PHONY: data
data: 
	@echo $(YELLOW)'\n'${NETWORK_NAME}": Downloading required data..."$(NOCOLOR)
	wget -nc -c https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names


.PHONY: clone_repo
clone_repo: 
	@echo $(YELLOW)'\n'${NETWORK_NAME}": Cloning the tensorflow yolo v3 repo..."$(NOCOLOR)
	@if [ -d ${TF_YOLO_REPO_NAME} ] ; \
	then \
		echo " - tensorflow yolo v3 repo already exists"; \
	else \
		echo " - cloning the tensorflow yolo v3 repo"; \
		${CLONE_REPO}; \
		cd tensorflow-yolo-v3; \
		git checkout ed60b9087b04e1d9ca40f8a9d8455d5c30c7c0d3; \
	fi 


.PHONY: convert_weights
convert_weights: clone_repo data
	@echo $(YELLOW)'\n'${NETWORK_NAME}": Converting weights...\n"$(NOCOLOR)
	wget -nc -c https://pjreddie.com/media/files/yolov3-tiny.weights; \
	cd tensorflow-yolo-v3; python3 convert_weights_pb.py --class_names ../coco.names --data_format NHWC --weights_file ../yolov3-tiny.weights --tiny; mv ${DARKNET_MODEL_BASE_NAME}.pb ../${NETWORK_NAME}.pb;


.PHONY: compile_model
compile_model: convert_weights
	@echo $(YELLOW)'\n'${NETWORK_NAME}": Compiling model to IR..."$(NOCOLOR)
	@echo "Checking OpenVINO environment..."
	@if [ -z "$(INTEL_OPENVINO_DIR)" ] ; \
	then \
		echo "Please initiate the Intel OpenVINO environment by going to the installation directory for openvino and running the setupvars.sh file in the bin folder." ; \
		exit 1 ; \
	else \
		echo "Intel OpenVINO environment is already set!" ; \
	fi
	@if [ -e ${NETWORK_NAME}.xml ] && [ -e ${NETWORK_NAME}.bin ] ; \
	then \
		echo " - IR files exist. No need to compile." ; \
	else \
	    ${NCCOMPILE} --input_model=${NETWORK_NAME}.pb --data_type=FP16 --batch=1 --tensorflow_use_custom_operations_config=$(INTEL_OPENVINO_DIR)/deployment_tools/model_optimizer/extensions/front/tf/yolo_v3_tiny.json; \
	fi


.PHONY: run_py
run_py: data deps compile_model
	@echo $(YELLOW)'\n'${NETWORK_NAME}": Running the python sample..."$(NOCOLOR)
	python3 ${NETWORK_NAME}.py --input=${INPUT} --ir=${IR} --labels=${LABELS}


.PHONY: run
run: run_py
	@echo $(YELLOW)'\n'${NETWORK_NAME}": Running the python sample..."$(NOCOLOR)



.PHONY: install-reqs
install-reqs: 
	@echo $(YELLOW)'\n'${NETWORK_NAME}": Checking application requirements..."$(NOCOLOR)
	@echo "No requirements needed."
	

.PHONY: uninstall-reqs
uninstall-reqs: 
	@echo $(YELLOW)'\n'${NETWORK_NAME}": Uninstalling requirements..."$(NOCOLOR)
	@echo "Nothing to uninstall."


.PHONY: help
help:
	@echo "\nPossible make targets: ";
	@echo $(YELLOW)"  make run or run_py "$(NOCOLOR)"- Runs the Python example.";
	@echo $(YELLOW)"  make help "$(NOCOLOR)"- Shows this message.";
	@echo $(YELLOW)"  make all "$(NOCOLOR)"- Makes everything needed to run, but doesn't run.";
	@echo $(YELLOW)"  make compile_model "$(NOCOLOR)"- Compiles the model to an IR format.";
	@echo $(YELLOW)"  make data "$(NOCOLOR)"- Downloads the required data.";
	@echo $(YELLOW)"  make deps "$(NOCOLOR)"- Makes dependencies for project, downloads and prepares model etc.";
	@echo $(YELLOW)"  make install-reqs "$(NOCOLOR)"- Installs requirements needed to run this sample on your system.";
	@echo $(YELLOW)"  make uninstall-reqs "$(NOCOLOR)"- Uninstalls requirements that were installed by the sample program.";
	@echo $(YELLOW)"  make clean "$(NOCOLOR)"- Removes all temporary files that are created by the Makefile.\n";
	@echo " ----- "$(YELLOW)"Model information"$(NOCOLOR)" -----"
	@echo $(YELLOW)" Description: "$(NOCOLOR)
	@echo "  This network does object detection on 80 different classes.";
	@echo $(YELLOW)" Model inputs:"$(NOCOLOR)
	@echo "  name: 'inputs', shape: [1x3x416x416], Expected color order is BGR after optimization.";
	@echo $(YELLOW)" Model outputs:"$(NOCOLOR)
	@echo "  name: 'detector/yolo-v3-tiny/Conv_12/BiasAdd/YoloRegion', shape: [1, 255, 26, 26]. See Readme for more details on the output.";
	@echo "  name: 'detector/yolo-v3-tiny/Conv_9/BiasAdd/YoloRegion', shape: [1, 255, 13, 13]. See Readme for more details on the output.\n";


clean:
	@echo $(YELLOW)'\n'${NETWORK_NAME}": CLeaning up files..."$(NOCOLOR)
	rm -f ${NETWORK_NAME}.xml
	rm -f ${NETWORK_NAME}.bin
	rm -f ${NETWORK_NAME}.mapping
	rm -f ${LABELS}
	rm -f ${WEIGHTS_FILE}
	rm -rf ${TF_YOLO_REPO_NAME}
	rm -f ${NETWORK_NAME}.pb
	
	
