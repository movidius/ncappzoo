GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
NOCOLOR = '\033[0m'


# Filename for the graph file
SSD_GESTURE_DIR = ssd_inception_v2_mo_fp16
SSD_GESTURE_BIN_FILENAME = frozen_inference_graph.bin
SSD_GESTURE_MAPPING_FILENAME = frozen_inference_graph.mapping
SSD_GESTURE_XML_FILENAME = frozen_inference_graph.xml
SSD_GESTURE_LABELS_FILENAME = frozen_inference_graph.labels

MODEL_RELATIVE_DIR = ../../tensorflow/SSD_Inception_V2/
MODEL_FP16_RELATIVE_DIR = ${MODEL_RELATIVE_DIR}/ssd_inception_v2_mo_fp16
PROJECT_RELATIVE_DIR = ../../apps/gesture_piarm/.

CPU_EXTENSION = $(INTEL_OPENVINO_DIR)/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so


.PHONY: all
all: deps data

.PHONY: data
data:
	@echo $(YELLOW)"\nGesture PiArm: Making data - no data needed."$(NOCOLOR)

.PHONY: deps
deps:
	@echo $(YELLOW)"\nGesture PiArm: Making dependencies..."$(NOCOLOR)
	@echo "Downloading .xml, .bin, and .mapping files..."
	@if [ -e ${SSD_GESTURE_DIR}/${SSD_GESTURE_BIN_FILENAME} ] ; \
	then \
		echo "SSD Inception V2 for Gesture Detection exists, skipping network request." ; \
	else \
		mkdir ${SSD_GESTURE_DIR} ; \
		python3 download_ir_model.py ; \
	fi
	@echo "Checking if OpenVINO environment is set..."
	@if [ -z "$(INTEL_OPENVINO_DIR)" ] ; \
	then \
		echo "Please initiate the Intel OpenVINO environment by going to the installation directory for openvino and running the setupvars.sh file in the bin folder." ; \
		exit 1 ; \
	else \
		echo "Intel OpenVINO environment is already set!" ; \
	fi


.PHONY: install-reqs
install-reqs:
	@echo $(YELLOW)"\nGesture Piarm: Checking application requirements...\n"$(NOCOLOR)
	@echo "Installing Pigpio and it's python library..."
	sudo apt-get install pigpio
	sudo pip3 install pigpio


.PHONY: run_wout_arm_CPU
run_wout_arm_CPU: deps data
	@echo $(YELLOW)"\nGesture PiArm: Running the sample without robot on CPU..."$(NOCOLOR)
	@echo "Starting Gesture SSD Python Demo..."
	python3 gesture_ssd_async_wout_robot.py -i 'cam' -m ${SSD_GESTURE_DIR}/${SSD_GESTURE_XML_FILENAME} -l ${CPU_EXTENSION}


.PHONY: run_wout_arm_MYRIAD
run_wout_arm_MYRIAD: install-reqs deps data
	@echo $(YELLOW)"\nGesture PiArm: Running the sample without robot using MYRIAD plugin..."$(NOCOLOR)
	@echo "Starting Gesture SSD Python Demo..."
	python3 gesture_ssd_async_wout_robot.py -i 'cam' -d MYRIAD -m ${SSD_GESTURE_DIR}/${SSD_GESTURE_XML_FILENAME}


.PHONY: run
run: install-reqs deps data
	@echo $(YELLOW)"\nGesture PiArm: Running the sample..."$(NOCOLOR)
	@echo "Initializing Pigpio daemon..."
	sudo pigpiod
	@echo "Starting Gesture SSD Python Demo..."
	python3 gesture_ssd_async.py -i 'cam' -d MYRIAD -m ${SSD_GESTURE_DIR}/${SSD_GESTURE_XML_FILENAME}
	@echo "Stopping Pigpio daemon..."
	sudo killall pigpiod


.PHONY: help
help:
	@echo $(YELLOW)"\nPossible make targets: ";
	@echo $(YELLOW)"  make run "$(NOCOLOR)"- runs the touchcalc.py python example program";
	@echo $(YELLOW)"  make help "$(NOCOLOR)"- shows this message";
	@echo $(YELLOW)"  make all "$(NOCOLOR)"- makes everything needed to run but doesn't run";
	@echo $(YELLOW)"  make clean "$(NOCOLOR)"- removes all created content in this directory."
	@echo $(YELLOW)"  make data "$(NOCOLOR)"- downloads data as needed"
	@echo $(YELLOW)"  make deps "$(NOCOLOR)"- makes/prepares dependencies"


.PHONY: clean
clean:
	@echo $(YELLOW)"\nGesture PiArm: Cleaning up files..."$(NOCOLOR)
	rm -rf ${SSD_GESTURE_DIR}
