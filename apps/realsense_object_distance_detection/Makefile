GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
NOCOLOR = '\033[0m'


APP_NAME = realsense_object_distance_detection
SSD_RELATIVE_DIRECTORY = ../../networks/ssd_mobilenet_v1_caffe
APP_RELATIVE_DIR = ../../apps/realsense_object_distance_detection
SSD_IR = mobilenet-ssd

.PHONY: all
all: deps data compile_cpp


.PHONY: data
data: 
	@echo $(YELLOW)'\n'${APP_NAME}": No data required."$(NOCOLOR)


.PHONY: deps
deps: default_model
	@echo $(YELLOW)'\n'${APP_NAME}": Making dependencies..."$(NOCOLOR)
	

.PHONY: default_model
default_model: 
	@echo $(YELLOW)'\n'${APP_NAME}": Making default models..."$(NOCOLOR)
	(cd ${SSD_RELATIVE_DIRECTORY}; make all; cp ${SSD_IR}.xml ${APP_RELATIVE_DIR}/; cp ${SSD_IR}.bin ${APP_RELATIVE_DIR}/;)


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
	@echo $(YELLOW)"  make default_model "$(NOCOLOR)"- compiles a default model to use when running";
	@echo $(YELLOW)"  make clean "$(NOCOLOR)"- removes all created content\n"
	@echo " ------ "$(YELLOW)"Keybindings"$(NOCOLOR)" -------"
	@echo "  q or Q - quits the program"
	@echo "  d or D - displays the depth sensor locations"
	@echo "  s or S - decreases the depth sensor location scale"
	@echo "  a or A - increases the depth sensor location scale\n"
	

clean: 
	@echo $(YELLOW)'\n'${APP_NAME}": Cleaning up files..."$(NOCOLOR);
	rm -rf build
	rm -f ${SSD_IR}.xml
	rm -f ${SSD_IR}.bin
	
