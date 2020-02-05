GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
NOCOLOR = '\033[0m'


APP_NAME = realsense_segmentation

ZOO_RELATIVE_DIR = ../../omz
MODEL_DOWNLOADER_DIR = open_model_zoo/tools/downloader
MODEL_DOWNLOADER_FILENAME = downloader.py
SEG_MODEL_ZOO_DIR = Transportation/segmentation/semantic_segmentation/icnet_icv/dldt/FP16
MODEL_FILE_NAME_BASE = semantic-segmentation-adas-0001

.PHONY: all
all: deps data compile_cpp


.PHONY: data
data: 
	@echo $(YELLOW)'\n'${APP_NAME}": No data required."$(NOCOLOR)


.PHONY: deps
deps: get_ir
	@echo $(YELLOW)'\n'${APP_NAME}": Making dependencies..."$(NOCOLOR)
	

.PHONY: model_zoo
model_zoo:
	@echo $(YELLOW)'\n'${NETWORK_NAME}": Cloning model zoo..."$(NOCOLOR)
	(cd ${ZOO_RELATIVE_DIR}; make all;) 


.PHONY: get_ir
get_ir:	model_zoo
	@echo $(YELLOW)'\n'${NETWORK_NAME}": Downloading IRs..."$(NOCOLOR);
	@if [ -e ${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}/${SEG_MODEL_ZOO_DIR}/${MODEL_FILE_NAME_BASE}.xml ] || [ -e ${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}/${SEG_MODEL_ZOO_DIR}/${MODEL_FILE_NAME_BASE}.bin ]; \
	then \
		echo " - Model files already exists in open model directory."; \
	else \
		echo " - Model files do not exist in open model directory. ";\
		echo " - Using Model downloader to download the model..."; \
		(cd ${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}; python3 ${MODEL_DOWNLOADER_FILENAME} --name ${MODEL_FILE_NAME_BASE};); \
	fi
	
	@if [ -e ${MODEL_FILE_NAME_BASE}.xml ] && [ -e ${MODEL_FILE_NAME_BASE}.bin ]; \
	then \
		echo " - IR files already exist in project directory. No need to copy IRs."; \
	else \
		echo " - IR files do not exist in project directory.";\
		echo " - Copying IR files to current folder..."; \
		cp ${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}/${SEG_MODEL_ZOO_DIR}/${MODEL_FILE_NAME_BASE}.xml .; \
		cp ${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}/${SEG_MODEL_ZOO_DIR}/${MODEL_FILE_NAME_BASE}.bin .; \
	fi


.PHONY:	compile_cpp
compile_cpp:  
	@echo $(YELLOW)'\n'${APP_NAME}": Building the sample..."$(NOCOLOR)
	@echo "Checking OpenVINO environment..."
	@if [ -z "$(INTEL_OPENVINO_DIR)" ] ; \
	then \
		echo "Please initiate the Intel OpenVINO environment by going to the installation directory for openvino and running the setupvars.sh file in the bin folder." ; \
		exit 1 ; \
	else \
		echo "Intel OpenVINO environment is already set!" ; \
	fi
	@echo "Checking Realsense prerequisites..."
	@if dpkg -l | grep librealsense2-dkms >> /dev/null ; \
	then \
	    echo "\nlibrealsense2-dkms is installed." ; \
	else \
	    echo $(YELLOW)"\nlibrealsense2-dkms is not installed. Please run the command 'make install-reqs' to install all sample requirements.\n"$(NOCOLOR); \
	    exit 1; \
	fi
	@if dpkg -l | grep librealsense2-dev >> /dev/null ; \
	then \
	    echo "\nlibrealsense2-dev is installed." ; \
	else \
	    echo $(YELLOW)"\nlibrealsense2-dev is not installed. Please run the command 'make install-reqs' to install all sample requirements.\n"$(NOCOLOR); \
	    exit 1; \
	fi
	(mkdir -p build; cd build; cmake ..; make;)


.PHONY: run
run: run_cpp
	

.PHONY: run_cpp
run_cpp: deps data compile_cpp
	@echo $(YELLOW)'\n'${APP_NAME}": Running the sample..."$(NOCOLOR)
	@echo "Checking Realsense prerequisites..."
	@if dpkg -l | grep librealsense2-dkms >> /dev/null ; \
	then \
	    echo "\nlibrealsense2-dkms is installed." ; \
	else \
	    echo $(YELLOW)"\nlibrealsense2-dkms is not installed. Please run the command 'make install-reqs' to install all sample requirements.\n"$(NOCOLOR); \
	    exit 1; \
	fi
	@if dpkg -l | grep librealsense2-dev >> /dev/null ; \
	then \
	    echo "\nlibrealsense2-dev is installed." ; \
	else \
	    echo $(YELLOW)"\nlibrealsense2-dev is not installed. Please run the command 'make install-reqs' to install all sample requirements.\n"$(NOCOLOR); \
	    exit 1; \
	fi
	(cd build; ./${APP_NAME}; cd ..;)


.PHONY: install-reqs
install-reqs: 
	@echo $(YELLOW)"\n"$(APP_NAME)": Checking installation requirements..."$(NOCOLOR)
	@echo "Installing Intel Realsense SDK 2 packages - dkms, dev..."
	@echo 'deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo xenial main' | sudo tee /etc/apt/sources.list.d/realsense-public.list; \
	sudo apt-key adv --keyserver keys.gnupg.net --recv-key 6F3EFCDE; \
	sudo apt-get update; \
	sudo apt-get install librealsense2-dkms; \
	sudo apt-get install librealsense2-dev


.PHONY: uninstall-reqs
uninstall-reqs: 
	@echo $(YELLOW)'\n'${APP_NAME}": Uninstalling requirements..."$(NOCOLOR)
	sudo apt-get -y purge librealsense2-dkms; \
	sudo apt-get -y purge librealsense2-dev
		

.PHONY: help
help:
	@echo "\nPossible make targets: ";
	@echo $(YELLOW)"  make run or run_cpp"$(NOCOLOR)"- runs the application";
	@echo $(YELLOW)"  make help "$(NOCOLOR)"- shows this message";
	@echo $(YELLOW)"  make all "$(NOCOLOR)"- makes everything needed to run but doesn't run";
	@echo $(YELLOW)"  make data "$(NOCOLOR)"- downloads data as needed";
	@echo $(YELLOW)"  make deps "$(NOCOLOR)"- makes/prepares dependencies";
	@echo $(YELLOW)"  make install-reqs "$(NOCOLOR)"- Installs requirements needed to run this sample on your system.";
	@echo          "        - This command will install the Realsense SDK 2.0 dkms and dev packages."
	@echo $(YELLOW)"  make uninstall-reqs "$(NOCOLOR)"- Uninstalls requirements that were installed by the sample program.";
	@echo $(YELLOW)"  make clean "$(NOCOLOR)"- removes all created content\n"
	@echo " ------ "$(YELLOW)"Keybindings"$(NOCOLOR)" -------"
	@echo "  q or Q - quits the program"
	

clean: 
	@echo $(YELLOW)'\n'${APP_NAME}": Cleaning up files..."$(NOCOLOR);
	rm -rf build
	rm -f ${MODEL_FILE_NAME_BASE}.bin
	rm -f ${MODEL_FILE_NAME_BASE}.xml
	
