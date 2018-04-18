
NCCOMPILE = mvNCCompile
NCPROFILE = mvNCProfile
NCCHECK   = mvNCCheck


NETWORK_SCRIPT_FILENAME= mnist_deep.py
GET_NETWORK_SCRIPT = wget -P . https://raw.githubusercontent.com/tensorflow/tensorflow/r1.4/tensorflow/examples/tutorials/mnist/${NETWORK_SCRIPT_FILENAME}

MODEL_DATA_FILENAME = mnist_model.data-00000-of-00001
MODEL_INDEX_FILENAME = mnist_model.index
MODEL_META_FILENAME = mnist_model.meta

MODEL_INFERENCE_DATA_FILENAME = mnist_inference.data-00000-of-00001
MODEL_INFERENCE_INDEX_FILENAME = mnist_inference.index
MODEL_INFERENCE_META_FILENAME = mnist_inference.meta


PATCH_FOR_NCS_FILENAME = orig_2_ncs.patch
PATCH_FOR_INFERENCE_FILENAME = ncs_2_inference.patch

MNIST_NCS_FILENAME = mnist_deep_mod.py
MNIST_INFERENCE_FILENAME = mnist_deep_inference.py

GET_MODEL_DATA = wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/neural_nets/master/tensorflow/mnist/${MODEL_INFERENCE_DATA_FILENAME}
GET_MODEL_INDEX = wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/neural_nets/master/tensorflow/mnist/${MODEL_INFERENCE_INDEX_FILENAME}
GET_MODEL_META =  wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/neural_nets/master/tensorflow/mnist/${MODEL_INFERENCE_META_FILENAME}

GRAPH_FILENAME = mnist_inference.graph

.PHONY: all
all: deps compile

.PHONY: deps
deps: model
	@echo "\nmaking deps"

.PHONY: network_code
network_code:
	@echo "\making network_code"
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
	fi


.PHONY: model
model:
	@echo "\nmaking model"
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
	@echo "\nmaking train";
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
	@echo "    only version of the network that the NCSDK can compile in these files:";
	@echo "      ${MODEL_INFERENCE_DATA_FILENAME}";
	@echo "      ${MODEL_INFERENCE_INDEX_FILENAME}";
	@echo "      ${MODEL_INFERENCE_META_FILENAME}";
	@echo "  "
	@echo "After complete, you will be able to compile the network you just trained with:"
	@echo "make compile"
	@echo ""
	@read -p"Press ENTER to continue the steps above (this will take about 20 min) or Ctrl-C to cancel" variable_nps;echo;

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


	@echo "patching example for NCSDK";
	patch ${NETWORK_SCRIPT_FILENAME} -i ${PATCH_FOR_NCS_FILENAME} -o ${MNIST_NCS_FILENAME};
	patch ${MNIST_NCS_FILENAME} -i ${PATCH_FOR_INFERENCE_FILENAME} -o ${MNIST_INFERENCE_FILENAME};
	@echo "Running patched code to start training";
	python3 ./${MNIST_NCS_FILENAME}
	@echo "Running code to save for NCSDK";
	python3 ./${MNIST_INFERENCE_FILENAME}


.PHONY: profile
profile: model
	@echo "\nmaking profile"
	@echo "No profile for this network yet"

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
compile: model
	@echo "\nmaking compile"
	@if [ -e ${GRAPH_FILENAME} ] ; \
	then \
		echo "graph file ${GRAPH_FILENAME} already exists, skipping compile"; \
	else \
		echo "Compiling to create ${GRAPH_FILENAME}"; \
		${NCCOMPILE} ${MODEL_INFERENCE_META_FILENAME} -w mnist_inference -s 12 -in input -on output -o ${GRAPH_FILENAME}; \
		if [ -e ${GRAPH_FILENAME} ] ; \
		then \
			echo "compile done, created ${GRAPH_FILENAME}."; \
		else \
			echo "***\nError - Could not compile, not able to create ${GRAPH_FILENAME} \n***\n"; \
			exit 1; \
		fi ; \
	fi


.PHONY: check
check: model
	@echo "\nmaking check"
	@echo "\No check for this network yet."


.PHONY: run
run: deps compile
	@echo "\nmaking run"
	python3 ./run.py

.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - makes the following: deps, compile";
	@echo "  make browse_profile - (TBD) runs the SDK profiler tool and brings up report in browser.";
	@echo "  make check - (TBD) runs SDK checker tool to verify an NCS graph file";
	@echo "  make clean - removes all created content"
	@echo "  make compile - runs SDK compiler tool to compile the graph file for the network";
	@echo "  make deps - downloads and prepares the trained network";
	@echo "  make profile - (TBD) runs the SDK profiler tool to profile the network creating output_report.html";
	@echo "  make run - runs the run.py python example program";
	@echo "  make train - creates NCSDK compatible version of TF network and trains it.";


clean: clean
	@echo "\nmaking clean"
	rm -f ${GRAPH_FILENAME}
	rm -f ${MODEL_INFERENCE_META_FILENAME}
	rm -f ${MODEL_INFERENCE_INDEX_FILENAME}
	rm -f ${MODEL_INFERENCE_DATA_FILENAME}
	rm -f ${NETWORK_SCRIPT_FILENAME}
	rm -f ${MODEL_META_FILENAME}
	rm -f ${MODEL_INDEX_FILENAME}
	rm -f ${MODEL_DATA_FILENAME}
	rm -f checkpoint
	rm -f ${MNIST_NCS_FILENAME}
	rm -f ${MNIST_INFERENCE_FILENAME}
