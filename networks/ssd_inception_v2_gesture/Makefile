NCCOMPILE = mo.py
CPU_EXTENSION = $(INTEL_OPENVINO_DIR)/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so

YELLOW='\033[1;33m'
NOCOLOR='\033[0m'
RED = '\033[1;31m'


GRAPH_FILENAME_BASE_IN_DIR_FP16 = frozen_inference_graph.xml
MO_LABELS = frozen_inference_graph.labels
MO_LABELS_IN_DIR_FP16 = frozen_inference_graph.labels
MO_INPUT_GRAPH_ARG = --input_model frozen_inference_graph.pb
MO_INPUT_PIPELINE = --tensorflow_object_detection_api_pipeline_config pipeline.config
MO_INPUT_CUSTOM_OPS = --tensorflow_use_custom_operations_config ssd_v2_support_inception.json
ZIPPED_LABELS = gesture_detection_model_labels.zip
MODEL_PB = frozen_inference_graph.pb
GET_MO_MODEL_LABELS = wget -c --no-cache https://downloadmirror.intel.com/29611/eng/gesture_detection_model_labels.zip
RUN_PY_RELATIVE_DIR = ssd_inception_v2_gesture.py


.PHONY: all
all: deps data compile_model

.PHONY: deps
deps: get_model
	@echo $(YELLOW)"\nSSD Inception V2 Gesture: Making dependencies... No extra dependencies needed."$(NOCOLOR)


.PHONY: data
data:


.PHONY: get_model
get_model:
	@if [ -e ${MODEL_PB} ] ; \
	then \
		echo "model file ${MODEL_PB} already exists, skipping download"; \
	else \
		echo "Downloading ${MODEL_PB} file" ; \
		${GET_MO_MODEL_LABELS} ; \
		unzip ${ZIPPED_LABELS} ; \
		if [ -e ${MODEL_PB} ] ; \
		then \
			echo "download ${MODEL_PB} done. Downloaded to /ssd_inception_v2_mo_fp16."; \
		else \
			echo "***\nError - Could not download ${MODEL_PB}. Check network and proxy settings \n***\n"; \
			exit 1; \
		fi ; \
	fi


.PHONY: train
train:
	@echo $(YELLOW) "\nSSD Inception V2 Geseture: Training model... NOT YET IMPLEMENTED" $(NOCOLOR);


.PHONY: compile_model
compile_model: get_model
	@echo $(YELLOW)"\nSSD Inception V2 Gesture: Compiling model to IR..."$(NOCOLOR)
	@if [ -e ${GRAPH_FILENAME_BASE_IN_DIR_FP16} ] ; \
	then \
		echo "compiled FP16 model file already exists, skipping compile."; \
	else \
		echo "Compiling FP16 model..."; \
		${NCCOMPILE} --data_type=FP16 --reverse_input_channels ${MO_INPUT_GRAPH_ARG} ${MO_INPUT_PIPELINE} ${MO_INPUT_CUSTOM_OPS} || (echo $(RED)"Make sure to set the OpenVINO environment variables using the "$(YELLOW)"setupvars.sh"$(RED)" script found in <your OpenVINO install location>/bin/ folder."$(NOCOLOR);  exit 1;); \
	fi


.PHONY: run_FP16
run_FP16: deps data compile_model
	@echo $(YELLOW) "\nSSD Inception V2: Running Python sample..." $(NOCOLOR)
	python3 $(RUN_PY_RELATIVE_DIR) -i 'cam' -m ${GRAPH_FILENAME_BASE_IN_DIR_FP16} --labels ${MO_LABELS_IN_DIR_FP16} -d MYRIAD


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
	@echo $(YELLOW)"  make all " $(NOCOLOR) "- Makes everything needed to run, but doesn't run the sample.";
	@echo $(YELLOW)"  make compile_model " $(NOCOLOR) "- Runs model compiler for the network.";
	@echo $(YELLOW)"  make get_model " $(NOCOLOR) "- Downloads the model files.";
	@echo $(YELLOW)"  make data " $(NOCOLOR) "- Downloads required data.";
	@echo $(YELLOW)"  make deps " $(NOCOLOR) "- Makes dependencies for project, prepares model etc.";
	@echo $(YELLOW)"  make install-reqs "$(NOCOLOR)"- Installs requirements needed to run this sample on your system.";
	@echo $(YELLOW)"  make uninstall-reqs "$(NOCOLOR)"- Uninstalls requirements that were installed by the sample program.";
	@echo $(YELLOW)"  make clean " $(NOCOLOR) "- Removes all the temporary files that are created by the Makefile.";
	@echo $(YELLOW)"  make train " $(NOCOLOR) "- Train a SSD Inception V2  model.";

clean: 
	@echo $(YELLOW)"\nSSD Inception V2 Gesture: Cleaning up files..."$(NOCOLOR)
	rm -f ${MO_LABELS_IN_DIR_FP16}
	rm -f ${MODEL_PB}
	rm -f ${GRAPH_FILENAME_BASE_IN_DIR_FP16}
	rm -f frozen_inference_graph.bin
	rm -f ${ZIPPED_LABELS}
	rm -f pipeline.config
	rm -f frozen_inference_graph.mapping