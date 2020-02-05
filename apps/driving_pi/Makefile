NCCOMPILE = mo.py

GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
RED = '\033[1;31m'
NOCOLOR = '\033[0m'

ZOO_RELATIVE_DIR = ../../omz
MODEL_DOWNLOADER_DIR = open_model_zoo/tools/downloader
MODEL_DOWNLOADER_FILENAME = downloader.py
ROAD_SEG_MODEL_ZOO_DIR = Transportation/segmentation/curbs/dldt/FP16
ROAD_SEG_MODEL_FILE_NAME_BASE = road-segmentation-adas-0001
ROAD_SEG_DESTINATION_DIR = src/data/road-segmentation

CLONE_TL_NET_REPO = git clone https://github.com/ayoobnabwani/ncappzoo_net.git
TL_NET_REPO_NAME = ncappzoo_net
TL_NET_DESTINATION_DIR = src/data/traffic-light

ROAD_SEG_NETWORK_NAME = road_segmentation_adas_0001
GAME_NAME = driving_pi
CROP =
CROPPED_ROAD_SEG_DIR =
SHOW = --show=yes

arch := $(shell arch)
CHECK_PI = python3 -c "from car_pi import DriveBrickPi3"

.PHONY: all
all: deps data clone_tl_repo


.PHONY: deps
deps: get_ir
	@echo $(YELLOW)'\n'${ROAD_SEG_NETWORK_NAME}": Making dependencies..."$(NOCOLOR)
	@if ${shell} ${CHECK_PI} 2> /dev/null ;\
	then \
		echo "BrickPi installed."; \
	else \
		echo ${YELLOW}"Dexter Industries BrickPi libraries are not installed. Please go to https://github.com/DexterInd/BrickPi to install."; \
		exit 1; \
	fi;


.PHONY: data
data:
	@echo $(YELLOW)'\n'${ROAD_SEG_NETWORK_NAME}": Downloading required data..."$(NOCOLOR)


.PHONY: model_zoo
model_zoo:
	@echo $(YELLOW)'\n'${ROAD_SEG_NETWORK_NAME}": Cloning model zoo..."$(NOCOLOR)
	(cd ${ZOO_RELATIVE_DIR}; make all;)


.PHONY: clone_tl_repo
clone_tl_repo:
	@echo $(YELLOW)'\n'${APP_NAME}": Cloning the traffic light repo..."$(NOCOLOR)
	@if [ -d ${TL_NET_REPO_NAME} ] ; \
	then \
		echo " - traffic-light network repo already exists"; \
	else \
		echo " - cloning the traffic-light network repo"; \
		${CLONE_TL_NET_REPO}; \
		mv ${TL_NET_REPO_NAME}/public_networks/traffic-light ${TL_NET_DESTINATION_DIR};\
	fi


.PHONY: get_ir
get_ir:	model_zoo
	@echo $(YELLOW)'\n'${ROAD_SEG_NETWORK_NAME}": Downloading IRs..."$(NOCOLOR);
	@if [ -e ${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}/${ROAD_SEG_MODEL_ZOO_DIR}/${ROAD_SEG_MODEL_FILE_NAME_BASE}.xml ] || [ -e ${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}/${ROAD_SEG_MODEL_ZOO_DIR}/${ROAD_SEG_MODEL_FILE_NAME_BASE}.bin ]; \
	then \
		echo " - Model files already exists in open model directory."; \
	else \
		echo " - Model files do not exist in open model directory. ";\
		echo " - Using Model downloader to download the model..."; \
		(cd ${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}; python3 ${MODEL_DOWNLOADER_FILENAME} --name ${ROAD_SEG_MODEL_FILE_NAME_BASE};); \
	fi

	@if [ -e ${ROAD_SEG_DESTINATION_DIR}/${ROAD_SEG_MODEL_FILE_NAME_BASE}.xml ] && [ -e ${ROAD_SEG_DESTINATION_DIR}/${ROAD_SEG_MODEL_FILE_NAME_BASE}.bin ]; \
	then \
		echo " - IR files already exist in project directory. No need to copy IRs."; \
	else \
		echo " - IR files do not exist in project directory.";\
		echo " - Copying IR files to current folder..."; \
		cp ${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}/${ROAD_SEG_MODEL_ZOO_DIR}/${ROAD_SEG_MODEL_FILE_NAME_BASE}.xml ${ROAD_SEG_DESTINATION_DIR}; \
		cp ${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}/${ROAD_SEG_MODEL_ZOO_DIR}/${ROAD_SEG_MODEL_FILE_NAME_BASE}.bin ${ROAD_SEG_DESTINATION_DIR}; \
	fi


.PHONY: compile_model
compile_model:
	@echo $(YELLOW)'\n'${ROAD_SEG_NETWORK_NAME}": Compiling model to IR..."$(NOCOLOR)
	@echo " - No compiling needed. Using Pre-compiled IRs from the open model zoo."


.PHONY: run
run: arch run_py


.PHONY: run_py
run_py: arch deps data clone_tl_repo
	@echo $(YELLOW)'\n'${GAME_NAME}": Running Python sample..."$(NOCOLOR)
	@echo "Checking OpenVINO environment..."
	@if [ -z "$(INTEL_OPENVINO_DIR)" ] ; \
	then \
		echo "Please initiate the Intel OpenVINO environment by going to the installation directory for openvino and running the setupvars.sh file in the bin folder." ; \
		exit 1 ; \
	else \
		echo "Intel OpenVINO environment is already set!" ; \
	fi
	python3 ${GAME_NAME}.py -fps -vr -s


.PHONY: install-reqs
install-reqs:
	@echo $(YELLOW)'\n'${ROAD_SEG_NETWORK_NAME}": Checking installation requirements..."$(NOCOLOR)
	@echo "No requirements needed."

.PHONY: uninstall-reqs
uninstall-reqs:
	@echo $(YELLOW)'\n'${ROAD_SEG_NETWORK_NAME}": Uninstalling requirements..."$(NOCOLOR)
	@echo "Nothing to uninstall."

.PHONY: arch
arch:
	@echo $(YELLOW)"Checking processor architecture..."$(NOCOLOR)
	@echo $(arch)
	@if [ "$(arch)" = "x86_64" ] ;\
	then \
		echo $(RED)"This app is only compatible with the Raspberry Pi.";\
		exit 1; \
	fi

.PHONY: help
help:
	@echo "\n Possible make targets: ";
	@echo $(YELLOW)"  make run or make run_py "$(NOCOLOR)"- Runs the Python example.";
	@echo $(YELLOW)"  make help "$(NOCOLOR)"- Shows this message";
	@echo $(YELLOW)"  make all "$(NOCOLOR)"- Makes everything needed to run, but doesn't run.";
	@echo $(YELLOW)"  make compile_model "$(NOCOLOR)"- Compiles the model to an IR format.";
	@echo $(YELLOW)"  make data "$(NOCOLOR)"- Downloads the required data.";
	@echo $(YELLOW)"  make clone_tl_net "$(NOCOLOR)"- Downloads traffic-light IR data.";
	@echo $(YELLOW)"  make deps "$(NOCOLOR)"- Makes dependencies for project, downloads and prepares model etc.";
	@echo $(YELLOW)"  make install-reqs "$(NOCOLOR)"- Installs requirements needed to run this sample on your system.";
	@echo $(YELLOW)"  make uninstall-reqs "$(NOCOLOR)"- Uninstalls requirements that were installed by the sample program.";
	@echo $(YELLOW)"  make clean "$(NOCOLOR)"- Removes all temporary files that are created by the Makefile.\n";
	@echo " ----- "$(YELLOW) $(ROAD_SEG_NETWORK_NAME) "Model information"$(NOCOLOR)" -----"
	@echo $(YELLOW)" Description: "$(NOCOLOR)
	@echo " This is a segmentation network to classify each pixel into four classes: BG, road, curb, mark.";
	@echo $(YELLOW)" Model inputs:"$(NOCOLOR)
	@echo " A blob with a BGR image in the format: [B, C=3, H=512, W=896]";
	@echo $(YELLOW)" Model outputs:"$(NOCOLOR)
	@echo " The output is a blob with the shape [B, C=4, H=512, W=896]. It can be treated as a four-channel feature map, where each channel is a probability of one of the classes: BG, road, curb, mark.\n";

clean:
	@echo $(YELLOW)'\n'${ROAD_SEG_NETWORK_NAME}": Cleaning up files..."$(NOCOLOR)
	rm -f ${ROAD_SEG_DESTINATION_DIR}/${ROAD_SEG_MODEL_FILE_NAME_BASE}.bin
	rm -f ${ROAD_SEG_DESTINATION_DIR}/${ROAD_SEG_MODEL_FILE_NAME_BASE}.xml
	rm -f -r ncappzoo_net
	rm -f -r src/data/traffic-light
