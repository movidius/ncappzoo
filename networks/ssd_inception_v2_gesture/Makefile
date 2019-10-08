NCCOMPILE = mo.py
CPU_EXTENSION = $(INTEL_OPENVINO_DIR)/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so

YELLOW='\033[1;33m'
NOCOLOR='\033[0m'
RED = '\033[1;31m'

GRAPH_FILENAME_BASE = ssd_inception_v2_food
GRAPH_DIR_FP32 = ssd_inception_v2_mo_fp32
GRAPH_DIR_FP16 = ssd_inception_v2_mo_fp16
GRAPH_FILENAME_BASE_IN_DIR_FP32 = ${GRAPH_DIR_FP32}/frozen_inference_graph.xml
GRAPH_FILENAME_BASE_IN_DIR_FP16 = ${GRAPH_DIR_FP16}/frozen_inference_graph.xml
MO_LABELS = frozen_inference_graph.labels
MO_LABELS_IN_DIR_FP32 = ${GRAPH_DIR_FP32}/frozen_inference_graph.labels
MO_LABELS_IN_DIR_FP16 = ssd_inception_v2_mo_fp16/frozen_inference_graph.labels
MO_INPUT_GRAPH_ARG = --input_model tensorflow_model/frozen_inference_graph.pb
MO_INPUT_PIPELINE = --tensorflow_object_detection_api_pipeline_config tensorflow_model/pipeline.config
MO_INPUT_CUSTOM_OPS = --tensorflow_use_custom_operations_config ssd_v2_support_inception.json
MO_OUTPUT_DIR_FP32 = --output_dir ${GRAPH_DIR_FP32}
MO_OUTPUT_DIR_FP16 = --output_dir ${GRAPH_DIR_FP16}

MODEL_DIR = tensorflow_model
MODEL_PIPELINE_CONFIG = ${MODEL_DIR}/pipeline.config
MODEL_PB = ${MODEL_DIR}/frozen_inference_graph.pb
MODEL_INFERENCE_PIPELINE_CONFIG = pipeline.config
MODEL_INFERENCE_PB = frozen_inference_graph.pb

GET_MODEL_PIPELINE = wget -c --no-cache -P ./tensorflow_model https://raw.githubusercontent.com/fcr3/gesture_detection/master/tensorflow_model/${MODEL_INFERENCE_PIPELINE_CONFIG}
GET_MO_MODEL_FP32_LABELS = wget -c --no-cache -P ./ssd_inception_v2_mo_fp32 https://raw.githubusercontent.com/fcr3/gesture_detection/master/model_optimized_fp32/${MO_LABELS}
GET_MO_MODEL_FP16_LABELS = wget -c --no-cache -P ./ssd_inception_v2_mo_fp16 https://raw.githubusercontent.com/fcr3/gesture_detection/master/model_optimized_fp16/${MO_LABELS}
GET_MODEL_PB = python3 download_weights.py

RUN_PY_RELATIVE_DIR = ssd_inception_v2_gesture.py


.PHONY: all
all: deps data compile_model

.PHONY: deps
deps: get_model
	@echo $(YELLOW)"\nSSD Inception V2 Gesture: Making dependencies... No extra dependencies needed."$(NOCOLOR)


.PHONY: data
data:
	@echo $(YELLOW)"\nSSD Inception V2 Gesture: Downloading required data... No data needed."$(NOCOLOR)


.PHONY: get_model
get_model:
	@echo $(YELLOW)"\nSSD Inception V2 Gesture: Downloading model files..."$(NOCOLOR)
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
	@echo $(YELLOW) "\nSSD Inception V2 Geseture: Training model... NOT YET IMPLEMENTED" $(NOCOLOR);


.PHONY: compile_model
compile_model: get_model
	@echo $(YELLOW)"\nSSD Inception V2 Gesture: Compiling model to IR..."$(NOCOLOR)
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


.PHONY: run_FP16
run_FP16: deps data compile_model
	@echo $(YELLOW) "\nSSD Inception V2: Running Python sample..." $(NOCOLOR)
	python3 $(RUN_PY_RELATIVE_DIR) -i 'cam' -m ${GRAPH_FILENAME_BASE_IN_DIR_FP16} --labels ${MO_LABELS_IN_DIR_FP16} -d MYRIAD


.PHONY: run_FP32
run_FP32: deps data compile_model
	@echo $(YELLOW) "\nSSD Inception V2: Running Python sample..." $(NOCOLOR)
	python3 $(RUN_PY_RELATIVE_DIR) -i 'cam' -m ${GRAPH_FILENAME_BASE_IN_DIR_FP32} -l ${CPU_EXTENSION} --labels ${MO_LABELS_IN_DIR_FP32}


.PHONY: run
run: run_py


.PHONY: run_py
run_py: deps data compile_model run_FP16


.PHONY: install-reqs
install-reqs:
	@echo $(YELLOW)"\nSSD Inception V2 Gesture: Checking application requirements...\n"$(NOCOLOR)
	@echo "No requirements needed."
	
	
.PHONY: uninstall-reqs
uninstall-reqs: 
	@echo $(YELLOW)"\nSSD Inception V2 Gesture: Uninstalling requirements..."$(NOCOLOR)
	@echo "Nothing to uninstall."


.PHONY: help
help:
	@echo "\nPossible make targets"
	@echo $(YELLOW)"  make help " $(NOCOLOR) "- Shows this message.";
	@echo $(YELLOW)"  make run or run_py " $(NOCOLOR) "- Runs the Python example.";
	@echo $(YELLOW)"  make run_FP16 "  $(NOCOLOR) "- Runs the Python example with FP16 model.";
	@echo $(YELLOW)"  make run_FP32 " $(NOCOLOR) "- Runs the Python example with FP32 model.";
	@echo $(YELLOW)"  make all " $(NOCOLOR) "- Makes everything needed to run, but doesn't run the sample.";
	@echo $(YELLOW)"  make compile_model " $(NOCOLOR) "- Runs model compiler for the network.";
	@echo $(YELLOW)"  make get_model " $(NOCOLOR) "- Downloads the model files.";
	@echo $(YELLOW)"  make data " $(NOCOLOR) "- Downloads required data.";
	@echo $(YELLOW)"  make deps " $(NOCOLOR) "- Makes dependencies for project, prepares model etc.";
	@echo $(YELLOW)"  make install-reqs "$(NOCOLOR)"- Installs requirements needed to run this sample on your system.";
	@echo $(YELLOW)"  make uninstall-reqs "$(NOCOLOR)"- Uninstalls requirements that were installed by the sample program.";
	@echo $(YELLOW)"  make clean " $(NOCOLOR) "- Removes all the temporary files that are created by the Makefile.";
	@echo $(YELLOW)"  make train " $(NOCOLOR) "- Train a SSD Inception V2  model.";

clean: clean
	@echo $(YELLOW)"\nSSD Inception V2 Gesture: Cleaning up files..."$(NOCOLOR)
	rm -rf ${GRAPH_DIR_FP32}
	rm -rf ${GRAPH_DIR_FP16}
	rm -rf ${MODEL_DIR}
