GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
NOCOLOR = '\033[0m'

# EDIT THESE EDAMAM API VARIABLES
APP_ID =
APP_KEY =

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

CPU_EXT = $(INTEL_OPENVINO_DIR)/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so


.PHONY: all
all: deps data
	@echo "Please install requirements by entering 'make install-reqs'"

.PHONY: data
data:
	@echo $(YELLOW)"\nChef Ai: Making data..."$(NOCOLOR)
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
	@echo $(YELLOW)"\nChef Ai: Making FP32 dependencies..."$(NOCOLOR)
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
	@echo $(YELLOW)"\nChef Ai: Making FP16 dependencies..."$(NOCOLOR)
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
	sudo pip3 install flask Pillow
	@echo "If you would like to uninstall the required packages, please enter 'make uninstall-reqs'"
	@echo "Checking if there a non-empty Edamam API app ID and app key..."
	@if [ -z "$(APP_KEY)" ] ; \
	then \
		echo "APP_KEY = $(APP_KEY)... Please paste or type in your Edamam app key into the APP_KEY variable located at the top of the Makefile." ; \
	else \
		echo "Provided a non-empty app key." ; \
	fi
	@if [ -z "$(APP_ID)" ] ; \
	then \
		echo "APP_ID = $(APP_ID)... Please paste or type in your Edamam app ID into the APP_ID variable located at the top of the Makefile." ; \
	else \
		echo "Provided a non-empty app ID." ; \
	fi


.PHONY: run
run: deps data
	@echo $(YELLOW)"\nChef Ai: Running the sample..."$(NOCOLOR)
	@echo "Please make sure to fill in the config values for the Edamam API!"
	@echo "Checking OpenVINO environment..."
	@if [ -z "$(INTEL_OPENVINO_DIR)" ] ; \
	then \
		echo "Please initiate the Intel OpenVINO environment by going to the installation directory for openvino and running the setupvars.sh file in the bin folder." ; \
		exit 1 ; \
	else \
		echo "Intel OpenVINO environment is already set!" ; \
	fi
	@echo "Checking if there a non-empty Edamam API app ID and app key..."
	@if [ -z "$(APP_KEY)" ] ; \
	then \
		echo "APP_KEY = $(APP_KEY)... Please paste or type in your Edamam app key into the APP_KEY variable located at the top of the Makefile." ; \
		exit 1 ; \
	else \
		echo "Provided a non-empty app key." ; \
	fi
	@if [ -z "$(APP_ID)" ] ; \
	then \
		echo "APP_ID = $(APP_ID)... Please paste or type in your Edamam app ID into the APP_ID variable located at the top of the Makefile." ; \
		exit 1 ; \
	else \
		echo "Provided a non-empty app ID." ; \
	fi
	@echo "Checking requirements..."
	@if python3 -c 'import pkgutil; exit(not pkgutil.find_loader("flask"))' ; \
	then \
		echo 'Flask found.' ; \
	else \
		echo "Flask not found. Please install requirements by entering 'make install-reqs'." ; \
		exit 1 ; \
	fi
	@if python3 -c 'import pkgutil; exit(not pkgutil.find_loader("PIL"))' ; \
	then \
		echo 'PIL found.' ; \
	else \
		echo "PIL not found. Please install requirements by entering 'make install-reqs'." ; \
		exit 1 ; \
	fi
	@echo "Starting backend server..."
	@python3 chef_ai_server.py -e ${CPU_EXT} -k ${APP_KEY} -i ${APP_ID}


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
	@echo $(YELLOW)"  make uninstall-reqs "$(NOCOLOR)"- uninstall required packages"


.PHONY: clean
clean:
	@echo $(YELLOW)"\nChef Ai: Cleaning up files..."$(NOCOLOR)
	rm -rf ${SSD_FOOD_DIR}
	rm -rf ${SSD_FOOD_DIR_CPU}
	rm -rf __pycache__
	rm -rf test_images

.PHONY: uninstall-reqs
uninstall-reqs:
	@echo $(YELLO)"\nChef App: Uninstalling flask and PIL..."$(NOCOLOR)
	@if python3 -c 'import pkgutil; exit(not pkgutil.find_loader("flask"))' ; \
	then \
		echo 'Flask found. Uninstalling now...' ; \
		sudo pip3 uninstall Flask ; \
	else \
		echo 'Flask not found. Not uninstalling package due to missing requirement.' ; \
	fi
	@if python3 -c 'import pkgutil; exit(not pkgutil.find_loader("PIL"))' ; \
	then \
		echo 'PIL found. Uninstalling now...' ; \
		sudo pip3 uninstall Pillow; \
	else \
		echo 'PIL not found. Not uninstalling package due to missing requirement.' ; \
	fi
