NCCOMPILE = mo.py

YELLOW='\033[1;33m'
NOCOLOR='\033[0m'
RED = '\033[1;31m'

NETWORK_SCRIPT_FILENAME= mnist_deep.py
GET_NETWORK_SCRIPT = wget -P . https://raw.githubusercontent.com/tensorflow/tensorflow/r1.4/tensorflow/examples/tutorials/mnist/${NETWORK_SCRIPT_FILENAME}
PATCH_FOR_NCS_FILENAME = orig_2_ncs.patch
PATCH_FOR_INFERENCE_FILENAME = ncs_2_inference.patch

MNIST_NCS_FILENAME = mnist_deep_mod.py
MNIST_INFERENCE_FILENAME = mnist_deep_inference.py

GRAPH_FILENAME_BASE = mnist

MODEL_DATA_FILENAME = mnist_model.data-00000-of-00001
MODEL_INDEX_FILENAME = mnist_model.index
MODEL_META_FILENAME = mnist_model.meta

MODEL_INFERENCE_DATA_FILENAME = mnist_inference.data-00000-of-00001
MODEL_INFERENCE_INDEX_FILENAME = mnist_inference.index
MODEL_INFERENCE_META_FILENAME = mnist_inference.meta

GET_MODEL_DATA = wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/neural_nets/master/tensorflow/mnist/${MODEL_INFERENCE_DATA_FILENAME}
GET_MODEL_INDEX = wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/neural_nets/master/tensorflow/mnist/${MODEL_INFERENCE_INDEX_FILENAME}
GET_MODEL_META =  wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/neural_nets/master/tensorflow/mnist/${MODEL_INFERENCE_META_FILENAME}

GRAPH_FILENAME_BASE = mnist_inference
MODEL_RELATIVE_DIR = .

RUN_PY_RELATIVE_DIR = mnist.py
TEST_IMAGE_RELATIVE_DIR = ../../data/digit_images
TEST_IMAGE_FILENAME = one.png
INPUT_SHAPE = [1,784]

.PHONY: all
all: deps data compile_model


.PHONY: deps
deps: get_model
	@echo $(YELLOW)"\nMnist: Making dependencies..."$(NOCOLOR)


.PHONY: data
data:
	@echo $(YELLOW)"\nMnist: Downloading required data - No data needed."$(NOCOLOR)


.PHONY: get_model
get_model:
	@echo $(YELLOW)"\nMnist: Downloading model files..."$(NOCOLOR)
	@if [ -e ${MODEL_INFERENCE_DATA_FILENAME} ] ; \
	then \
		echo "model file ${MODEL_INFERENCE_DATA_FILENAME} already exists, skipping download"; \
	else \
		echo "Downloading ${MODEL_INFERENCE_DATA_FILENAME} file"; \
		${GET_MODEL_DATA}; \
		if [ -e ${MODEL_INFERENCE_DATA_FILENAME} ] ; \
		then \
			echo "download ${MODEL_INFERENCE_DATA_FILENAME} done."; \
		else \
			echo "***\nError - Could not download ${MODEL_INFERENCE_DATA_FILENAME}. Check network and proxy settings \n***\n"; \
			exit 1; \
		fi ; \
	fi
	@if [ -e ${MODEL_INFERENCE_META_FILENAME} ] ; \
	then \
		echo "model file ${MODEL_INFERENCE_META_FILENAME} already exists, skipping download"; \
	else \
		echo "Downloading ${MODEL_INFERENCE_META_FILENAME} file"; \
		${GET_MODEL_META}; \
		if [ -e ${MODEL_INFERENCE_META_FILENAME} ] ; \
		then \
			echo "download ${MODEL_INFERENCE_META_FILENAME} done."; \
		else \
			echo "***\nError - Could not download ${MODEL_INFERENCE_META_FILENAME}. Check network and proxy settings \n***\n"; \
			exit 1; \
		fi ; \
	fi
	@if [ -e ${MODEL_INFERENCE_INDEX_FILENAME} ] ; \
	then \
		echo "model file ${MODEL_INFERENCE_INDEX_FILENAME} already exists, skipping download"; \
	else \
		echo "Downloading ${MODEL_INFERENCE_INDEX_FILENAME} file"; \
		${GET_MODEL_INDEX}; \
		if [ -e ${MODEL_INFERENCE_INDEX_FILENAME} ] ; \
		then \
			echo "download ${MODEL_INFERENCE_INDEX_FILENAME} done."; \
		else \
			echo "***\nError - Could not download ${MODEL_INFERENCE_INDEX_FILENAME}. Check network and proxy settings \n***\n"; \
			exit 1; \
		fi ; \
	fi
	
	
.PHONY: train
train:
	@echo $(YELLOW) "\nMnist: Training model..." $(NOCOLOR);
	@echo "\nTo train the network the following steps will be taken:";
	@echo "  - Download the TensorFlow mnist_deep.py example";
	@echo "  - Modify the mnist_deep.py -> mnist_deep_mod.py for NCS compatibility";
	@echo "  - run mnist_deep_mod.py to train the network, it will save trained network in:";
	@echo "      ${MODEL_DATA_FILENAME}";
	@echo "      ${MODEL_INDEX_FILENAME}";
	@echo "      ${MODEL_META_FILENAME}";
	@echo "  - modify mnist_deep_mod.py -> mnist_deep_inference.py for inference only version";
	@echo "    of the network (no training code)."
	@echo "  - run mnist_deep_inference.py to read the saved network and write an inference";
	@echo "    only version of the network that the OpenVINO Model Optimizer can compile in these files:";
	@echo "      ${MODEL_INFERENCE_DATA_FILENAME}";
	@echo "      ${MODEL_INFERENCE_INDEX_FILENAME}";
	@echo "      ${MODEL_INFERENCE_META_FILENAME}";
	@echo "  "
	@echo "After complete, you will be able to compile the network you just trained with:";
	@echo "make compile_model";
	@echo ""
	@read -p"Press ENTER to continue the steps above (this will take about 20 min) or Ctrl-C to cancel" variable_nps; echo;

	@if [ -e ${NETWORK_SCRIPT_FILENAME} ] ; \
	then \
		echo "network script ${NETWORK_SCRIPT_FILENAME} already exists, skipping download"; \
	else \
		echo "Downloading ${NETWORK_SCRIPT_FILENAME} file"; \
		${GET_NETWORK_SCRIPT}; \
		if [ -e ${NETWORK_SCRIPT_FILENAME} ] ; \
		then \
			echo "download ${NETWORK_SCRIPT_FILENAME} done."; \
		else \
			echo "***\nError - Could not download ${NETWORK_SCRIPT_FILENAME}. Check network and proxy settings \n***\n"; \
			exit 1; \
		fi ; \
	fi ;

	@echo "patching example for the OpenVINO Toolkit";
	patch ${NETWORK_SCRIPT_FILENAME} -i ${PATCH_FOR_NCS_FILENAME} -o ${MNIST_NCS_FILENAME};
	patch ${MNIST_NCS_FILENAME} -i ${PATCH_FOR_INFERENCE_FILENAME} -o ${MNIST_INFERENCE_FILENAME};
	@echo "Running patched code to start training";
	python3 ./${MNIST_NCS_FILENAME}
	@echo "Running code to save for the OpenVINO Toolkit";
	python3 ./${MNIST_INFERENCE_FILENAME}
	@echo $(YELLOW)"\nMnist model training complete."$(NOCOLOR)
	@echo "Model name: "${MODEL_INFERENCE_META_FILENAME} 
	@echo $(NOCOLOR)"\nRun the command "$(YELLOW)"make compile_model"$(NOCOLOR)" to compile your model or "$(YELLOW)"make run"$(NOCOLOR)" to run the sample with your trained model.\n"


.PHONY: compile_model
compile_model: get_model
	@echo $(YELLOW)"\nMnist: Compiling model to IR..."$(NOCOLOR)
	@if [ -e ${GRAPH_FILENAME_BASE}.xml ] ; \
	then \
		echo "compiled model file already exists, skipping compile."; \
	else \
		${NCCOMPILE} --data_type=FP16 --input_meta_graph=${MODEL_INFERENCE_META_FILENAME} --input_shape=${INPUT_SHAPE} || (echo $(RED)"Make sure to set the OpenVINO environment variables using the "$(YELLOW)"setupvars.sh"$(RED)" script found in <your OpenVINO install location>/bin/ folder."$(NOCOLOR);  exit 1;); \
	fi

	
.PHONY: run
run: run_py


.PHONY: run_py
run_py: deps data compile_model
	@echo $(YELLOW) "\nMnist: Running Python sample..." $(NOCOLOR)
	python3 $(RUN_PY_RELATIVE_DIR) --ir=${MODEL_RELATIVE_DIR}/${GRAPH_FILENAME_BASE}.xml --image=${TEST_IMAGE_RELATIVE_DIR}/${TEST_IMAGE_FILENAME} --labels=categories.txt


.PHONY: install-reqs
install-reqs: 
	@echo $(YELLOW)"\nMnist: Checking application requirements...\n"$(NOCOLOR)
	@echo "No requirements needed."
	
	
.PHONY: uninstall-reqs
uninstall-reqs: 
	@echo $(YELLOW)"\nMnist: Uninstalling requirements..."$(NOCOLOR)
	@echo "Nothing to uninstall."
	
		
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
	@echo $(YELLOW)"  make install-reqs "$(NOCOLOR)"- Installs requirements needed to run this sample on your system.";
	@echo $(YELLOW)"  make uninstall-reqs "$(NOCOLOR)"- Uninstalls requirements that were installed by the sample program.";
	@echo $(YELLOW)"  make clean " $(NOCOLOR) "- Removes all the temporary files that are created by the Makefile.";
	@echo $(YELLOW)"  make train " $(NOCOLOR) "- Train a mnist model.";

clean: clean
	@echo $(YELLOW)"\nMnist: Cleaning up files..."$(NOCOLOR)
	rm -f ${GRAPH_FILENAME_BASE}.bin
	rm -f ${GRAPH_FILENAME_BASE}.mapping.json
	rm -f ${GRAPH_FILENAME_BASE}.mapping
	rm -f ${GRAPH_FILENAME_BASE}.xml
	rm -f ${MODEL_INFERENCE_META_FILENAME}
	rm -f ${MODEL_INFERENCE_INDEX_FILENAME}
	rm -f ${MODEL_INFERENCE_DATA_FILENAME}
	rm -f ${MNIST_NCS_FILENAME}
	rm -f ${MNIST_INFERENCE_FILENAME}
	rm -f ${NETWORK_SCRIPT_FILENAME}
	rm -f ${MODEL_DATA_FILENAME}
	rm -f ${MODEL_INDEX_FILENAME}
	rm -f ${MODEL_META_FILENAME}
	rm -f checkpoint
	
