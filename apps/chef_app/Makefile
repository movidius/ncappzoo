GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
NOCOLOR = '\033[0m'


# Filename for the graph file
SSD_FOOD_DIR = ssd_inception_v2_mo_fp16
SSD_FOOD_DIR_CPU = ssd_inception_v2_mo_fp32
SSD_FOOD_BIN_FILENAME = frozen_inference_graph.bin
SSD_FOOD_MAPPING_FILENAME = frozen_inference_graph.mapping
SSD_FOOD_XML_FILENAME = frozen_inference_graph.xml
SSD_FOOD_LABELS_FILENAME = frozen_inference_graph.labels

MODEL_RELATIVE_DIR = ../../tensorflow/SSD_Inception_V2/
MODEL_FP16_RELATIVE_DIR = ${MODEL_RELATIVE_DIR}/ssd_inception_v2_mo_fp16
PROJECT_RELATIVE_DIR = ../../apps/gesture_piarm/.

CPU_EXTENSION = /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so

GET_MO_MODEL_FP32_XML = wget -c --no-cache -P ./ssd_inception_v2_mo_fp32 https://raw.githubusercontent.com/fcr3/chef_devmesh/master/food_model_fp32/${SSD_FOOD_XML_FILENAME}
GET_MO_MODEL_FP16_XML = wget -c --no-cache -P ./ssd_inception_v2_mo_fp16 https://raw.githubusercontent.com/fcr3/chef_devmesh/master/food_model_fp16/${SSD_FOOD_XML_FILENAME}
GET_MO_MODEL_FP32_BIN = wget -c --no-cache -P ./ssd_inception_v2_mo_fp32 https://raw.githubusercontent.com/fcr3/chef_devmesh/master/food_model_fp32/${SSD_FOOD_BIN_FILENAME}
GET_MO_MODEL_FP16_BIN = wget -c --no-cache -P ./ssd_inception_v2_mo_fp16 https://raw.githubusercontent.com/fcr3/chef_devmesh/master/food_model_fp16/${SSD_FOOD_BIN_FILENAME}
GET_MO_MODEL_FP32_MAPPING = wget -c --no-cache -P ./ssd_inception_v2_mo_fp32 https://raw.githubusercontent.com/fcr3/chef_devmesh/master/food_model_fp32/${SSD_FOOD_MAPPING_FILENAME}
GET_MO_MODEL_FP16_MAPPING = wget -c --no-cache -P ./ssd_inception_v2_mo_fp16 https://raw.githubusercontent.com/fcr3/chef_devmesh/master/food_model_fp16/${SSD_FOOD_MAPPING_FILENAME}
GET_MO_MODEL_FP32_LABELS = wget -c --no-cache -P ./ssd_inception_v2_mo_fp32 https://raw.githubusercontent.com/fcr3/chef_devmesh/master/food_model_fp32/${SSD_FOOD_LABELS_FILENAME}
GET_MO_MODEL_FP16_LABELS = wget -c --no-cache -P ./ssd_inception_v2_mo_fp16 https://raw.githubusercontent.com/fcr3/chef_devmesh/master/food_model_fp16/${SSD_FOOD_LABELS_FILENAME}

GET_TEST_IMG1 = wget -c --no-cache -P ./test_images https://raw.githubusercontent.com/fcr3/chef_devmesh/master/test_images/test1.JPG
GET_TEST_IMG2 = wget -c --no-cache -P ./test_images https://raw.githubusercontent.com/fcr3/chef_devmesh/master/test_images/test2.JPG
GET_TEST_IMG3 = wget -c --no-cache -P ./test_images https://raw.githubusercontent.com/fcr3/chef_devmesh/master/test_images/test3.JPG


.PHONY: all
all: deps data

.PHONY: data
data:
	@echo $(YELLOW)"\Chef App: Making data..."$(NOCOLOR)
	@echo $(YELLOW)"\nChef App: Downloading required data..."$(NOCOLOR)
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


.PHONY: deps
deps: deps_FP32 deps_FP16

.PHONY: deps_FP32
deps_FP32:
	@echo $(YELLOW)"\nChef App: Making FP32 dependencies..."$(NOCOLOR)
	@echo "Downloading .xml, .bin, and .mapping files..."
	@if [ -e ${SSD_FOOD_DIR_CPU}/${SSD_FOOD_BIN_FILENAME} ] ; \
	then \
		echo "SSD Inception V2 for Food Detection exists, skipping network request." ; \
	else \
		${GET_MO_MODEL_FP32_BIN} ; \
	fi
	@if [ -e ${SSD_FOOD_DIR_CPU}/${SSD_FOOD_XML_FILENAME} ] ; \
	then \
		echo "SSD Inception V2 for Food Detection exists, skipping network request." ; \
	else \
		${GET_MO_MODEL_FP32_XML} ; \
	fi
	@if [ -e ${SSD_FOOD_DIR_CPU}/${SSD_FOOD_LABELS_FILENAME} ] ; \
	then \
		echo "SSD Inception V2 for Food Detection exists, skipping network request." ; \
	else \
		${GET_MO_MODEL_FP32_LABELS} ; \
	fi
	@if [ -e ${SSD_FOOD_DIR_CPU}/${SSD_FOOD_MAPPING_FILENAME} ] ; \
	then \
		echo "SSD Inception V2 for Food Detection exists, skipping network request." ; \
	else \
		${GET_MO_MODEL_FP32_MAPPING} ; \
	fi


.PHONY: deps_FP16
deps_FP16:
	@echo $(YELLOW)"\nChef App: Making FP16 dependencies..."$(NOCOLOR)
	@echo "Downloading .xml, .bin, and .mapping files..."
	@if [ -e ${SSD_FOOD_DIR}/${SSD_FOOD_BIN_FILENAME} ] ; \
	then \
		echo "SSD Inception V2 for Food Detection exists, skipping network request." ; \
	else \
		${GET_MO_MODEL_FP16_BIN} ; \
	fi
	@if [ -e ${SSD_FOOD_DIR}/${SSD_FOOD_XML_FILENAME} ] ; \
	then \
		echo "SSD Inception V2 for Food Detection exists, skipping network request." ; \
	else \
		${GET_MO_MODEL_FP16_XML} ; \
	fi
	@if [ -e ${SSD_FOOD_DIR}/${SSD_FOOD_LABELS_FILENAME} ] ; \
	then \
		echo "SSD Inception V2 for Food Detection exists, skipping network request." ; \
	else \
		${GET_MO_MODEL_FP16_LABELS} ; \
	fi
	@if [ -e ${SSD_FOOD_DIR}/${SSD_FOOD_MAPPING_FILENAME} ] ; \
	then \
		echo "SSD Inception V2 for Food Detection exists, skipping network request." ; \
	else \
		${GET_MO_MODEL_FP16_MAPPING} ; \
	fi

.PHONY: install-reqs
install-reqs:
	@echo $(YELLOW)"\ngesture_piarm: Checking application requirements...\n"$(NOCOLOR)
	@echo "Installing Node.js and npm into the root..."
	curl -sL https://deb.nodesource.com/setup_10.x | sudo -E bash -
	sudo apt install nodejs
	@echo "Verifying Installation of node and npm..."
	@echo "Node version:"
	node --version
	@echo "NPM version:"
	npm --version
	@echo "Note: make clean will delete Node.js and npm completely. Please use make clean_light if you would like to keep Node.js or npm." 


.PHONY: run_wout_arm_CPU
run_wout_arm_CPU: deps data
	@echo $(YELLOW)"\ngesture_piarm: Running the sample without robot on CPU..."$(NOCOLOR)
	@echo "Setting up OpenVINO environment..."
	@/opt/intel/openvino/bin/setupvars.sh
	@echo "Starting Gesture SSD Python Demo..."
	python3 gesture_ssd_async_wout_robot2.py -i 'cam' -m ${SSD_GESTURE_DIR}/${SSD_GESTURE_XML_FILENAME} -l ${CPU_EXTENSION}


.PHONY: run_wout_arm_MYRIAD
run_wout_arm_MYRIAD: install-reqs deps data
	@echo $(YELLOW)"\ngesture_piarm: Running the sample without robot using MYRIAD plugin..."$(NOCOLOR)
	@echo "Setting up OpenVINO environment..."
	@/opt/intel/openvino/bin/setupvars.sh
	@echo "Starting Gesture SSD Python Demo..."
	python3 gesture_ssd_async_wout_robot.py -i 'cam' -d MYRIAD -m ${SSD_GESTURE_DIR}/${SSD_GESTURE_XML_FILENAME}


.PHONY: run
run: install-reqs deps data
	@echo $(YELLOW)"\ngesture_piarm: Running the sample..."$(NOCOLOR)
	@echo "Setting up OpenVINO environment..."
	/opt/intel/openvino/bin/setupvars.sh
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


.PHONY: clean_lite
clean:
	@echo $(YELLOW)"\ngesture_piarm: Cleaning up files..."$(NOCOLOR)
	rm -rf ${SSD_FOOD_DIR}
	rm -rf ${SSD_FOOD_DIR_CPU}
	rm -rf test_images

.PHONY: clean
clean: 
