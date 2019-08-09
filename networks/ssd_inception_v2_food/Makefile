NCCOMPILE = mo.py
CPU_EXTENSION = $(INTEL_OPENVINO_DIR)/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so

YELLOW='\033[1;33m'
NOCOLOR='\033[0m'
RED = '\033[1;31m'

GRAPH_FILENAME_BASE = ssd_inception_v2_grocery
GRAPH_DIR_FP32 = ssd_inception_v2_mo_fp32
GRAPH_DIR_FP16 = ssd_inception_v2_mo_fp16
GRAPH_FILENAME_BASE_IN_DIR_FP32 = ${GRAPH_DIR_FP32}/frozen_inference_graph.xml
GRAPH_FILENAME_BASE_IN_DIR_FP16 = ${GRAPH_DIR_FP16}/frozen_inference_graph.xml
MO_LABELS = frozen_inference_graph.labels
MO_LABELS_IN_DIR_FP32	= ${GRAPH_DIR_FP32}/frozen_inference_graph.labels
MO_LABELS_IN_DIR_FP16 = ${GRAPH_DIR_FP16}/frozen_inference_graph.labels
MO_INPUT_GRAPH_ARG = --input_model tensorflow_model/frozen_inference_graph.pb
MO_INPUT_PIPELINE = --tensorflow_object_detection_api_pipeline_config tensorflow_model/ssd_inception_v2_coco.config
MO_INPUT_CUSTOM_OPS = --tensorflow_use_custom_operations_config ssd_v2_support_inception.json
MO_OUTPUT_DIR_FP32 = --output_dir ssd_inception_v2_mo_fp32
MO_OUTPUT_DIR_FP16 = --output_dir ssd_inception_v2_mo_fp16

MODEL_DIR = tensorflow_model
MODEL_PIPELINE_CONFIG = tensorflow_model/ssd_inception_v2_coco.config
MODEL_PB = tensorflow_model/frozen_inference_graph.pb
MODEL_INFERENCE_PIPELINE_CONFIG = ssd_inception_v2_coco.config
MODEL_INFERENCE_PB = frozen_inference_graph.pb

GET_MODEL_PIPELINE = wget -c --no-cache -P ./tensorflow_model https://raw.githubusercontent.com/fcr3/chef_devmesh/master/tensorflow_model/${MODEL_INFERENCE_PIPELINE_CONFIG}
GET_MO_MODEL_FP32_LABELS = wget -c --no-cache -P ./ssd_inception_v2_mo_fp32 https://raw.githubusercontent.com/fcr3/chef_devmesh/master/food_model_fp32/${MO_LABELS}
GET_MO_MODEL_FP16_LABELS = wget -c --no-cache -P ./ssd_inception_v2_mo_fp16 https://raw.githubusercontent.com/fcr3/chef_devmesh/master/food_model_fp16/${MO_LABELS}
GET_TEST_IMG1 = wget -c --no-cache -P ./test_images https://raw.githubusercontent.com/fcr3/chef_devmesh/master/test_images/test1.JPG
GET_TEST_IMG2 = wget -c --no-cache -P ./test_images https://raw.githubusercontent.com/fcr3/chef_devmesh/master/test_images/test2.JPG
GET_TEST_IMG3 = wget -c --no-cache -P ./test_images https://raw.githubusercontent.com/fcr3/chef_devmesh/master/test_images/test3.JPG
GET_MODEL_PB = python3 download_weights.py

RUN_PY_RELATIVE_DIR = ssd_inception_v2_food.py
TEST_IMG_DIR = test_images
TEST_IMG1 = test_images/test1.JPG
TEST_IMG2 = test_images/test2.JPG
TEST_IMG3 = test_images/test3.JPG


.PHONY: all
all: deps data compile_model

.PHONY: deps
deps: get_model
	@echo $(YELLOW)"\nSSD Inception V2 Food: Making dependencies... No extra dependencies needed."$(NOCOLOR)


.PHONY: data
data:
	@echo $(YELLOW)"\nSSD Inception V2 Food: Downloading required data..."$(NOCOLOR)
	@if [ -e ${TEST_IMG1} ] ; \
	then \
		echo "test file ${TEST_IMG1} already exists, skipping download"; \
	else \
		echo "Downloading ${TEST_IMG1} file"; \
		${GET_TEST_IMG1}; \
	fi
	@if [ -e ${TEST_IMG2} ] ; \
	then \
		echo "test file ${TEST_IMG2} already exists, skipping download"; \
	else \
		echo "Downloading ${TEST_IMG2} file"; \
		${GET_TEST_IMG2}; \
	fi
	@if [ -e ${TEST_IMG3} ] ; \
	then \
		echo "test file ${TEST_IMG3} already exists, skipping download"; \
	else \
		echo "Downloading ${TEST_IMG3} file"; \
		${GET_TEST_IMG3}; \
	fi


.PHONY: get_model
get_model:
	@echo $(YELLOW)"\nSSD Inception V2 Food: Downloading model files..."$(NOCOLOR)
	@if [ -e ${MODEL_PIPELINE_CONFIG} ] ; \
	then \
		echo "model file ${MODEL_PIPELINE_CONFIG} already exists, skipping download"; \
	else \
		echo "Downloading ${MODEL_INFERENCE_PIPELINE_CONFIG} file"; \
		${GET_MODEL_PIPELINE}; \
		if [ -e ${MODEL_PIPELINE_CONFIG} ] ; \
		then \
			echo "download ${MODEL_INFERENCE_PIPELINE_CONFIG} done. Downloaded to /tensorflow_model."; \
		else \
			echo "***\nError - Could not download ${MODEL_INFERENCE_PIPELINE_CONFIG}. Check network and proxy settings \n***\n"; \
			exit 1; \
		fi ; \
	fi
	@if [ -e ${MODEL_PB} ] ; \
	then \
		echo "model file ${MODEL_PB} already exists, skipping download"; \
	else \
		echo "Downloading ${MODEL_INFERENCE_PB} file" ; \
		${GET_MODEL_PB} ; \
		if [ -e ${MODEL_PB} ] ; \
		then \
			echo "download ${MODEL_INFERENCE_PB} done. Downloaded to /tensorflow_model."; \
		else \
			echo "***\nError - Could not download ${MODEL_INFERENCE_PB}. Check network and proxy settings \n***\n"; \
			exit 1; \
		fi ; \
	fi


.PHONY: train
train:
	@echo $(YELLOW) "\nSSD Inception V2 Food: Training model... NOT YET IMPLEMENTED" $(NOCOLOR);


.PHONY: compile_model
compile_model: get_model
	@echo $(YELLOW)"\nSSD Inception V2 Food: Compiling model to IR..."$(NOCOLOR)
	@if [ -e ${GRAPH_FILENAME_BASE_IN_DIR_FP32} ] ; \
	then \
		echo "compiled FP32 model file already exists, skipping compile."; \
	else \
		echo "Compiling FP32 model..."; \
		mkdir ssd_inception_v2_mo_fp32 ; \
		${NCCOMPILE} --data_type=FP32 --reverse_input_channels ${MO_INPUT_GRAPH_ARG} ${MO_INPUT_PIPELINE} ${MO_INPUT_CUSTOM_OPS} ${MO_OUTPUT_DIR_FP32} || (echo $(RED)"Make sure to set the OpenVINO environment variables using the "$(YELLOW)"setupvars.sh"$(RED)" script found in <your OpenVINO install location>/bin/ folder."$(NOCOLOR);  exit 1;); \
	fi
	@if [ -e ${GRAPH_FILENAME_BASE_IN_DIR_FP16} ] ; \
	then \
		echo "compiled FP16 model file already exists, skipping compile."; \
	else \
		echo "Compiling FP16 model..."; \
		mkdir ssd_inception_v2_mo_fp16 ; \
		${NCCOMPILE} --data_type=FP16 --reverse_input_channels ${MO_INPUT_GRAPH_ARG} ${MO_INPUT_PIPELINE} ${MO_INPUT_CUSTOM_OPS} ${MO_OUTPUT_DIR_FP16} || (echo $(RED)"Make sure to set the OpenVINO environment variables using the "$(YELLOW)"setupvars.sh"$(RED)" script found in <your OpenVINO install location>/bin/ folder."$(NOCOLOR);  exit 1;); \
	fi
	@if [ -e ${MO_LABELS_IN_DIR_FP32} ] ; \
	then \
		echo "compiled FP32 model labels already exist, skipping network request."; \
	else \
		echo "Grabbing labels for FP32 model..."; \
		${GET_MO_MODEL_FP32_LABELS}; \
	fi
	@if [ -e ${MO_LABELS_IN_DIR_FP16} ] ; \
	then \
		echo "compiled FP16 model labels already exist, skipping network request."; \
	else \
		echo "Grabbing labels for FP16 model..."; \
		${GET_MO_MODEL_FP16_LABELS} ;\
	fi


.PHONY: install-reqs
install-reqs:
	@echo $(YELLOW)"\nSSD Inception V2 Food: Checking application requirements...\n"$(NOCOLOR)
	@echo "No requirements needed."


.PHONY: run_FP16
run_FP16: install-reqs deps data compile_model
	@echo $(YELLOW) "\nSSD Inception V2 Food: Running Python sample..." $(NOCOLOR)
	python3 $(RUN_PY_RELATIVE_DIR) -i ${TEST_IMG1} -m ${GRAPH_FILENAME_BASE_IN_DIR_FP16} -l ${MO_LABELS_IN_DIR_FP16} -d MYRIAD


.PHONY: run_FP32
run_FP32: install-reqs deps data compile_model
	@echo $(YELLOW) "\nSSD Inception V2 Food: Running Python sample..." $(NOCOLOR)
	python3 $(RUN_PY_RELATIVE_DIR) -i ${TEST_IMG1} -m ${GRAPH_FILENAME_BASE_IN_DIR_FP32} -e ${CPU_EXTENSION} --l ${MO_LABELS_IN_DIR_FP32}


.PHONY: run
run: run_py


.PHONY: run_py
run_py: install-reqs deps data compile_model run_FP16


.PHONY: help
help:
	@echo "\nPossible make targets"
	@echo $(YELLOW)"  make help " $(NOCOLOR) "- Shows this message.";
	@echo $(YELLOW)"  make run " $(NOCOLOR) "- Runs the Python example.";
	@echo $(YELLOW)"  make run_py " $(NOCOLOR) "- Runs the Python example.";
	@echo $(YELLOW)"  make all " $(NOCOLOR) "- Makes everything needed to run, but doesn't run the sample.";
	@echo $(YELLOW)"  make compile_model " $(NOCOLOR) "- Runs model compiler for the network.";
	@echo $(YELLOW)"  make model " $(NOCOLOR) "- Downloads the model files.";
	@echo $(YELLOW)"  make data " $(NOCOLOR) "- Downloads required data.";
	@echo $(YELLOW)"  make deps " $(NOCOLOR) "- Makes dependencies for project, prepares model etc.";
	@echo $(YELLOW)"  make clean " $(NOCOLOR) "- Removes all the temporary files that are created by the Makefile.";
	@echo $(YELLOW)"  make train " $(NOCOLOR) "- Train a mnist model.";

clean: clean
	@echo $(YELLOW)"\nSSD Inception V2 Food: Cleaning up files..."$(NOCOLOR)
	rm -rf ${GRAPH_DIR_FP32}
	rm -rf ${GRAPH_DIR_FP16}
	rm -rf ${MODEL_DIR}
	rm -rf ${TEST_IMG_DIR}
