
NCCOMPILE = mo.py

GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
RED = '\033[1;31m'
NOCOLOR = '\033[0m'

MODEL_BASE_NAME = frozen_darknet_yolov3_model

IMG = ../../data/images/nps_chair.png
REPO_NAME = tensorflow-yolo-v3
REPO = tensorflow-yolo-v3
CLONE_REPO = git clone https://github.com/mystic123/tensorflow-yolo-v3.git
LABELS = coco.names
WEIGHTS_FILE = yolov3-tiny.weights

.PHONY: all
all: data deps compile_model


.PHONY: deps
deps: 
	@echo $(YELLOW)"\nTiny Yolo v3: Making dependencies..."$(NOCOLOR)

	
.PHONY: data
data:
	@echo $(YELLOW)"\nTiny Yolo v3: Downloading required data..."$(NOCOLOR)
	wget -nc -c https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
	wget -nc -c https://pjreddie.com/media/files/yolov3-tiny.weights


.PHONY: clone_repo
clone_repo: 
	@echo $(YELLOW)"\nTiny Yolo v3: Downloading prototxt file..."$(NOCOLOR)
	@if [ -d ${REPO} ] ; \
	then \
		echo " - tensorflow yolo v3 repo already exists"; \
	else \
		echo " - cloning the tensorflow yolo v3 repo"; \
		${CLONE_REPO}; \
		cd tensorflow-yolo-v3; \
		git checkout ed60b9087b04e1d9ca40f8a9d8455d5c30c7c0d3; \
	fi 


.PHONY: convert_weights
convert_weights: data clone_repo
	@echo $(YELLOW)"\nTiny Yolo v3: Converting weights...\n"$(NOCOLOR)
	cd tensorflow-yolo-v3; python3 convert_weights_pb.py --class_names ../coco.names --data_format NHWC --weights_file ../yolov3-tiny.weights --tiny; 



.PHONY: install-reqs
install-reqs: 
	@echo $(YELLOW)"\nTiny Yolo v3: Checking application requirements...\n"$(NOCOLOR)
	@echo "No requirements needed."
	
	
.PHONY: compile_model
compile_model: convert_weights
	@echo $(YELLOW)"\nTiny Yolo v3: Compiling model to IR..."$(NOCOLOR)
	@if [ -e ${MODEL_BASE_NAME}.xml ] && [ -e ${MODEL_BASE_NAME}.bin ] ; \
	then \
		echo " - IR files exist. No need to compile." ; \
	else \
	    ${NCCOMPILE} --input_model=tensorflow-yolo-v3/frozen_darknet_yolov3_model.pb --data_type=FP16 --batch=1 --tensorflow_use_custom_operations_config=$(INTEL_OPENVINO_DIR)/deployment_tools/model_optimizer/extensions/front/tf/yolo_v3_tiny.json || (echo $(RED)"Make sure your paths are correct and the OpenVINO environment variables are set using the "$(YELLOW)"setupvars.sh"$(RED)" script found in <your OpenVINO install location>/bin/ folder."$(NOCOLOR);  exit 1;); \
	fi


.PHONY: run_py
run_py: install-reqs data deps compile_model
	@echo $(YELLOW)"\nTiny Yolo v3: Running the python sample..."$(NOCOLOR)
	python3 ./tiny_yolo_v3.py --image=${IMG} --ir=${IR}


.PHONY: run
run: run_py
	@echo $(YELLOW)"\nTiny Yolo v3: Running the python sample..."$(NOCOLOR)


.PHONY: help
help:
	@echo "\nPossible make targets: ";
	@echo $(YELLOW)"  make help "$(NOCOLOR)"- shows this message";
	@echo $(YELLOW)"  make run "$(NOCOLOR)"- runs the run.py python example program";
	@echo $(YELLOW)"  make all "$(NOCOLOR)"- makes the following: data deps compile_model";
	@echo $(YELLOW)"  make prototxt "$(NOCOLOR)"- downloads and adds input shape to Caffe prototxt file";
	@echo $(YELLOW)"  make caffemodel "$(NOCOLOR)"- downloads the caffemodel for the network"
	@echo $(YELLOW)"  make compile "$(NOCOLOR)"- runs Model Optimizer compiler tool to compile an IR file for the network";
	@echo $(YELLOW)"  make clean "$(NOCOLOR)"- removes all created content"





clean:
	@echo $(YELLOW)"\nTiny Yolo v3: making clean..."$(NOCOLOR)
	rm -f ${MODEL_BASE_NAME}.xml
	rm -f ${MODEL_BASE_NAME}.bin
	rm -f ${MODEL_BASE_NAME}.mapping
	rm -f ${LABELS}
	rm -f ${WEIGHTS_FILE}
	rm -rf ${REPO}
	
	

