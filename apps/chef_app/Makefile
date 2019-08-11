GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
NOCOLOR = '\033[0m'


# Filename for the graph file
SSD_FOOD_DIR = food_model_fp16
SSD_FOOD_DIR_CPU = food_model_fp32
SSD_FOOD_BIN_FILENAME = frozen_inference_graph.bin
SSD_FOOD_MAPPING_FILENAME = frozen_inference_graph.mapping
SSD_FOOD_XML_FILENAME = frozen_inference_graph.xml
SSD_FOOD_LABELS_FILENAME = frozen_inference_graph.labels

GET_MO_MODEL_FP32_XML = wget -c --no-cache -P ./food_model_fp32 https://raw.githubusercontent.com/fcr3/chef_devmesh/master/food_model_fp32/${SSD_FOOD_XML_FILENAME}
GET_MO_MODEL_FP16_XML = wget -c --no-cache -P ./food_model_fp16 https://raw.githubusercontent.com/fcr3/chef_devmesh/master/food_model_fp16/${SSD_FOOD_XML_FILENAME}
GET_MO_MODEL_FP32_BIN = wget -c --no-cache -P ./food_model_fp32 https://raw.githubusercontent.com/fcr3/chef_devmesh/master/food_model_fp32/${SSD_FOOD_BIN_FILENAME}
GET_MO_MODEL_FP16_BIN = wget -c --no-cache -P ./food_model_fp16 https://raw.githubusercontent.com/fcr3/chef_devmesh/master/food_model_fp16/${SSD_FOOD_BIN_FILENAME}
GET_MO_MODEL_FP32_MAPPING = wget -c --no-cache -P ./food_model_fp32 https://raw.githubusercontent.com/fcr3/chef_devmesh/master/food_model_fp32/${SSD_FOOD_MAPPING_FILENAME}
GET_MO_MODEL_FP16_MAPPING = wget -c --no-cache -P ./food_model_fp16 https://raw.githubusercontent.com/fcr3/chef_devmesh/master/food_model_fp16/${SSD_FOOD_MAPPING_FILENAME}
GET_MO_MODEL_FP32_LABELS = wget -c --no-cache -P ./food_model_fp32 https://raw.githubusercontent.com/fcr3/chef_devmesh/master/food_model_fp32/${SSD_FOOD_LABELS_FILENAME}
GET_MO_MODEL_FP16_LABELS = wget -c --no-cache -P ./food_model_fp16 https://raw.githubusercontent.com/fcr3/chef_devmesh/master/food_model_fp16/${SSD_FOOD_LABELS_FILENAME}

TEST_IMG1 = test_images/test1.JPG
TEST_IMG2 = test_images/test2.JPG
TEST_IMG3 = test_images/test3.JPG

GET_TEST_IMG1 = wget -c --no-cache -P ./test_images https://raw.githubusercontent.com/fcr3/chef_devmesh/master/test_images/test1.JPG
GET_TEST_IMG2 = wget -c --no-cache -P ./test_images https://raw.githubusercontent.com/fcr3/chef_devmesh/master/test_images/test2.JPG
GET_TEST_IMG3 = wget -c --no-cache -P ./test_images https://raw.githubusercontent.com/fcr3/chef_devmesh/master/test_images/test3.JPG


.PHONY: all
all: deps data

.PHONY: data
data:
	@echo $(YELLOW)"\nChef App: Making data..."$(NOCOLOR)
	@echo "Downloading required data..."
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
	@echo "Installing Flask and Pillow (PIL - Python Imaging Library)..."
	pip3 install flask Pillow


.PHONY: run
run: install-reqs deps data
	@echo $(YELLOW)"\nChef App: Running the sample..."$(NOCOLOR)
	@echo "Setting up OpenVINO environment..."
	/opt/intel/openvino/bin/setupvars.sh
	@echo "Starting backend server..."
	@gnome-terminal -e "python3 chef_ai_server.py"
	@echo "Opening up webpage..."
	@xdg-open http://127.0.0.1:5000


.PHONY: help
help:
	@echo $(YELLOW)"\nPossible make targets: ";
	@echo $(YELLOW)"  make run "$(NOCOLOR)"- runs the touchcalc.py python example program";
	@echo $(YELLOW)"  make help "$(NOCOLOR)"- shows this message";
	@echo $(YELLOW)"  make all "$(NOCOLOR)"- makes everything needed to run but doesn't run";
	@echo $(YELLOW)"  make clean "$(NOCOLOR)"- removes all created content in this directory."
	@echo $(YELLOW)"  make data "$(NOCOLOR)"- downloads data as needed"
	@echo $(YELLOW)"  make deps "$(NOCOLOR)"- makes/prepares dependencies"
	@echo $(YELLOW)"  make deps_FP16 "$(NOCOLOR)"- makes/prepares dependencies for the FP16 model"
	@echo $(YELLOW)"  make deps_FP32 "$(NOCOLOR)"- makes/prepares dependencies for the FP32 model"
	@echo $(YELLOW)"  make install-reqs "$(NOCOLOR)"- install required packages"


.PHONY: clean-lite
clean-lite:
	@echo $(YELLOW)"\nChef App: Cleaning up files..."$(NOCOLOR)
	rm -rf ${SSD_FOOD_DIR}
	rm -rf ${SSD_FOOD_DIR_CPU}
	rm -rf __pycache__
	rm -rf test_images

.PHONY: clean
clean: clean-lite
	@echo "Uninstalling flask and PIL..."
	pip3 uninstall flask Pillow
