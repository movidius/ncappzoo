
NCCOMPILE = mvNCCompile
NCPROFILE = mvNCProfile
NCCHECK   = mvNCCheck


NETWORK_SCRIPT_FILENAME= mnist_deep.py
GET_NETWORK_SCRIPT = wget -P . https://raw.githubusercontent.com/tensorflow/tensorflow/r1.4/tensorflow/examples/tutorials/mnist/${NETWORK_SCRIPT_FILENAME}

MODEL_INFERENCE_DATA_FILENAME = mnist_inference.data-00000-of-00001
MODEL_INFERENCE_INDEX_FILENAME = mnist_inference.index
MODEL_INFERENCE_META_FILENAME = mnist_inference.meta

IMAGE_DIR = ./digit_images
GET_IMAGES = mkdir -p ${IMAGE_DIR}; \
             wget -c --no-cache -P ${IMAGE_DIR} https://raw.githubusercontent.com/nealvis/media/master/digit_pics/licenses.txt; \
             wget -c --no-cache -P ${IMAGE_DIR} https://raw.githubusercontent.com/nealvis/media/master/digit_pics/zero.png; \
             wget -c --no-cache -P ${IMAGE_DIR} https://raw.githubusercontent.com/nealvis/media/master/digit_pics/one.png; \
             wget -c --no-cache -P ${IMAGE_DIR} https://raw.githubusercontent.com/nealvis/media/master/digit_pics/twp.png; \
             wget -c --no-cache -P ${IMAGE_DIR} https://raw.githubusercontent.com/nealvis/media/master/digit_pics/three.png; \
             wget -c --no-cache -P ${IMAGE_DIR} https://raw.githubusercontent.com/nealvis/media/master/digit_pics/four.png; \
             wget -c --no-cache -P ${IMAGE_DIR} https://raw.githubusercontent.com/nealvis/media/master/digit_pics/five.png; \
             wget -c --no-cache -P ${IMAGE_DIR} https://raw.githubusercontent.com/nealvis/media/master/digit_pics/six.png; \
             wget -c --no-cache -P ${IMAGE_DIR} https://raw.githubusercontent.com/nealvis/media/master/digit_pics/seven.png; \
             wget -c --no-cache -P ${IMAGE_DIR} https://raw.githubusercontent.com/nealvis/media/master/digit_pics/eight.png; \
             wget -c --no-cache -P ${IMAGE_DIR} https://raw.githubusercontent.com/nealvis/media/master/digit_pics/nine.png


GET_MODEL_DATA = wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/neural_nets/master/tensorflow/mnist/${MODEL_INFERENCE_DATA_FILENAME}
GET_MODEL_INDEX = wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/neural_nets/master/tensorflow/mnist/${MODEL_INFERENCE_INDEX_FILENAME}
GET_MODEL_META =  wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/neural_nets/master/tensorflow/mnist/${MODEL_INFERENCE_META_FILENAME}

GRAPH_FILENAME = mnist_inference.graph

.PHONY: all
all: deps compile

.PHONY: deps
deps: images model
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


.PHONY: images
images:
	@echo "\nmaking images"
	${GET_IMAGES};


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


.PHONY: train_network
train_network: network_code
	@echo "\nmaking train_network"


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
run: compile
	@echo "\nmaking run"
	python3 ./run.py

.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - makes the following: deps, compile";
	@echo "  make deps - downloads and prepares the trained network";
	@echo "  make compile - runs SDK compiler tool to compile the graph file for the network";
	@echo "  make check - (TBD) runs SDK checker tool to verify an NCS graph file";
	@echo "  make profile - (TBD) runs the SDK profiler tool to profile the network creating output_report.html";
	@echo "  make browse_profile - (TBD) runs the SDK profiler tool and brings up report in browser.";
	@echo "  make run - runs the run.py python example program";
	@echo "  make clean - removes all created content"


clean: clean
	@echo "\nmaking clean"
	rm -f ${GRAPH_FILENAME}
	rm -f ${MODEL_INFERENCE_META_FILENAME}
	rm -f ${MODEL_INFERENCE_INDEX_FILENAME}
	rm -f ${MODEL_INFERENCE_DATA_FILENAME}
	rm -f ${NETWORK_SCRIPT_FILENAME}
	rm -rf ${IMAGE_DIR}
