
NCCOMPILE = mvNCCompile
NCPROFILE = mvNCProfile
NCCHECK   = mvNCCheck

TEST_IMAGE_FILENAME = 0.jpg
GET_IMAGE = wget -P . https://raw.githubusercontent.com/ihere1/movidius-face/master/face_detect_pi/two_devices_input/${TEST_IMAGE_FILENAME}

MKDIR_P = mkdir -p
P2838_DIR = p2838
O_DIR = o
WGET = wget -c --no-cache -P
RAW_URL = https://raw.githubusercontent.com/ihere1/movidius-face/master/face_detect_pi/

MODEL_DATA_FILENAME = model.data-00000-of-00001
MODEL_INDEX_FILENAME = model.index
MODEL_META_FILENAME = model.meta

P2838_GRAPH_FILENAME = p2838.graph
O_GRAPH_FILENAME = o.graph

.PHONY: image
image:
	@echo "\nmaking image"
	@if [ -e ${TEST_IMAGE_FILENAME} ] ; \
	then \
		echo "model file ${TEST_IMAGE_FILENAME} already exists, skipping download"; \
	else \
		echo "Downloading ${TEST_IMAGE_FILENAME} file"; \
		${GET_IMAGE}; \
		if [ -e ${TEST_IMAGE_FILENAME} ] ; \
		then \
			echo "download ${TEST_IMAGE_FILENAME} done."; \
		else \
			echo "***\nError - Could not download ${TEST_IMAGE_FILENAME}. Check network and proxy settings \n***\n"; \
			exit 1; \
		fi ; \
	fi

.PHONY: download_net_file
download_net_file:
	${MKDIR_P} ${OUT_DIR}
	@if [ -e ${OUT_DIR}/${FILENAME} ] ; \
	then \
		echo "network script ${OUT_DIR}/${FILENAME} already exists, skipping download"; \
	else \
		echo "Downloading ${OUT_DIR}/${FILENAME} file"; \
		${WGET} ${OUT_DIR} ${RAW_URL}/${OUT_DIR}/${FILENAME}; \
		if [ -e ${OUT_DIR}/${FILENAME} ] ; \
		then \
			echo "download ${OUT_DIR}/${FILENAME} done."; \
		else \
			echo "***\nError - Could not download ${OUT_DIR}/${FILENAME}. Check network and proxy settings \n***\n"; \
			exit 1; \
		fi ; \
	fi

.PHONY: p2838net_data
p2838net_data: export FILENAME = ${MODEL_DATA_FILENAME}
p2838net_data: export OUT_DIR = ${P2838_DIR}
p2838net_data:
	make download_net_file

.PHONY: p2838net_index
p2838net_index: export FILENAME = ${MODEL_INDEX_FILENAME}
p2838net_index: export OUT_DIR = ${P2838_DIR}
p2838net_index:
	make download_net_file

.PHONY: p2838net_meta
p2838net_meta: export FILENAME = ${MODEL_META_FILENAME}
p2838net_meta: export OUT_DIR = ${P2838_DIR}
p2838net_meta:
	make download_net_file

.PHONY: onet_data
onet_data: export FILENAME = ${MODEL_DATA_FILENAME}
onet_data: export OUT_DIR = ${O_DIR}
onet_data:
	make download_net_file

.PHONY: onet_index
onet_index: export FILENAME = ${MODEL_INDEX_FILENAME}
onet_index: export OUT_DIR = ${O_DIR}
onet_index:
	make download_net_file

.PHONY: onet_meta
onet_meta: export FILENAME = ${MODEL_META_FILENAME}
onet_meta: export OUT_DIR = ${O_DIR}
onet_meta:
	make download_net_file

.PHONY: all
all: deps compile

.PHONY: deps
deps: image p2838net_data p2838net_index p2838net_meta onet_data onet_index onet_meta

.PHONY: profile
profile: deps
	@echo "\nmaking profile"
	@echo "No profile for this network yet"

.PHONY: check
check: deps
	@echo "\nmaking check"
	@echo "\No check for this network yet."

.PHONY: compile
compile: deps
	@echo "\nmaking compile"
	@if [ -e ${P2838_GRAPH_FILENAME} ] ; \
	then \
		echo "graph file ${P2838_GRAPH_FILENAME} already exists, skipping compile"; \
	else \
		echo "Compiling to create ${P2838_GRAPH_FILENAME}"; \
		${NCCOMPILE} -in pnet/input -on pnet/output -o ${P2838_GRAPH_FILENAME} ${P2838_DIR}/${MODEL_META_FILENAME}; \
		if [ -e ${P2838_GRAPH_FILENAME} ] ; \
		then \
			echo "compile done, created ${P2838_GRAPH_FILENAME}."; \
		else \
			echo "***\nError - Could not compile, not able to create ${P2838_GRAPH_FILENAME} \n***\n"; \
			exit 1; \
		fi ; \
	fi
	@if [ -e ${O_GRAPH_FILENAME} ] ; \
	then \
		echo "graph file ${O_GRAPH_FILENAME} already exists, skipping compile"; \
	else \
		echo "Compiling to create ${O_GRAPH_FILENAME}"; \
		${NCCOMPILE} -in onet/input -on onet/output -o ${O_GRAPH_FILENAME} ${O_DIR}/${MODEL_META_FILENAME}; \
		if [ -e ${O_GRAPH_FILENAME} ] ; \
		then \
			echo "compile done, created ${O_GRAPH_FILENAME}."; \
		else \
			echo "***\nError - Could not compile, not able to create ${O_GRAPH_FILENAME} \n***\n"; \
			exit 1; \
		fi ; \
	fi

.PHONY: run
run: deps compile
	@echo "\nmaking run"
	python3 ./run.py

.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - makes the following: deps, compile";
	@echo "  make deps - downloads and prepares the trained network";
	@echo "  make check - (TBD) runs SDK checker tool to verify an NCS graph file";
	@echo "  make compile - runs SDK compiler tool to compile the graph file for the network";
	@echo "  make profile - (TBD) runs the SDK profiler tool to profile the network creating output_report.html";
	@echo "  make run - runs the run.py python example program";
	@echo "  make clean - removes all created content"


clean: clean
	@echo "\nmaking clean"
	rm -f ${TEST_IMAGE_FILENAME}
	rm -rf ${P2838_DIR}
	rm -rf ${O_DIR}
	rm -f ${P2838_GRAPH_FILENAME}
	rm -f ${O_GRAPH_FILENAME}

