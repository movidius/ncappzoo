GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
NOCOLOR = '\033[0m'


APP_NAME = realsense_person_distance_detection
SSD_RELATIVE_DIRECTORY = ../../caffe/SSD_Mobilenet 
APP_RELATIVE_DIR = ../../apps/realsense_person_distance_detection
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
	(mkdir -p build; cd build; cmake ..; make;)


.PHONY: run
run: run_cpp
	

.PHONY: run_cpp
run_cpp: deps data compile_cpp
	@echo $(YELLOW)'\n'${APP_NAME}": Running the sample..."$(NOCOLOR)
	(cd build; ./${APP_NAME}; cd ..;)


.PHONY: install-reqs
install-reqs: 
	@echo $(YELLOW)"\n"$(APP_NAME)": Checking installation requirements..."$(NOCOLOR)
	@echo "No requirements needed."	


.PHONY: uninstall-reqs
uninstall-reqs: 
	@echo $(YELLOW)'\n'${APP_NAME}": Uninstalling requirements..."$(NOCOLOR)
	@echo "Nothing to uninstall."
	

.PHONY: help
help:
	@echo "\nPossible make targets: ";
	@echo $(YELLOW)"  make run or run_cpp"$(NOCOLOR)"- runs the application";
	@echo $(YELLOW)"  make help "$(NOCOLOR)"- shows this message";
	@echo $(YELLOW)"  make all "$(NOCOLOR)"- makes everything needed to run but doesn't run";
	@echo $(YELLOW)"  make data "$(NOCOLOR)"- downloads data as needed";
	@echo $(YELLOW)"  make deps "$(NOCOLOR)"- makes/prepares dependencies";
	@echo $(YELLOW)"  make install-reqs "$(NOCOLOR)"- Installs requirements needed to run this sample on your system.";
	@echo $(YELLOW)"  make uninstall-reqs "$(NOCOLOR)"- Uninstalls requirements that were installed by the sample program.";
	@echo $(YELLOW)"  make default_model "$(NOCOLOR)"- compiles a default model to use when running";
	@echo $(YELLOW)"  make clean "$(NOCOLOR)"- removes all created content"
	@echo ""


clean: 
	@echo $(YELLOW)'\n'${APP_NAME}": Cleaning up files..."$(NOCOLOR);
	rm -rf build
	rm -f ${SSD_IR}.xml
	rm -f ${SSD_IR}.bin
	
