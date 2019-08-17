
# filenames for the graph files that we'll copy to this directory.
CLASSIFIER_GRAPH_BASE_FILENAME = googlenet-v1
CLASSIFIER_RELATIVE_DIR = ../../caffe/GoogLeNet
XML = ${CLASSIFIER_GRAPH_BASE_FILENAME}.xml
BIN = ${CLASSIFIER_GRAPH_BASE_FILENAME}.bin
DEV = MYRIAD

# name of the directory and the base name of the main python file (minus the .py extension)
APP_NAME = benchmark_ncs
GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
NOCOLOR = '\033[0m'


.PHONY: all
all: data deps

.PHONY: compile_default_model
compile_default_model:
	@echo $(YELLOW)'\n'${APP_NAME}": Compiling the model"$(NOCOLOR)
	@if [ ${XML} != "googlenet-v1.xml" ] && [ ${BIN} != "googlenet-v1.bin" ]; \
	then \
		echo " - User has provided IR. Skipping compiling Googlenet."; \
	else \
		if [ -e ${CLASSIFIER_GRAPH_BASE_FILENAME}.xml ] && [ -e ${CLASSIFIER_GRAPH_BASE_FILENAME}.bin ] ; \
		then \
			echo "${CLASSIFIER_GRAPH_BASE_FILENAME}.xml file already exists, skipping compiliation"; \
		else \
			(cd ${CLASSIFIER_RELATIVE_DIR}; make compile_model; cd ../../apps/${APP_NAME}; cp ${CLASSIFIER_RELATIVE_DIR}/${CLASSIFIER_GRAPH_BASE_FILENAME}.xml . ; cp ${CLASSIFIER_RELATIVE_DIR}/${CLASSIFIER_GRAPH_BASE_FILENAME}.bin . ; ); \
		fi \
	fi


.PHONY: default_model
default_model: compile_default_model
	@echo $(YELLOW)'\n'${APP_NAME}": Making the default model..."$(NOCOLOR)


.PHONY: data
data: 
	@echo $(YELLOW)'\n'${APP_NAME}": No data needed."$(NOCOLOR)


.PHONY: deps
deps: default_model
	@echo $(YELLOW)'\n'${APP_NAME}": Making dependencies..."$(NOCOLOR)


.PHONY: run_py
run_py: data deps
	@echo $(YELLOW)'\n'${APP_NAME}": Running the sample..."$(NOCOLOR)
	python3 ./${APP_NAME}.py dev=${DEV} mx=${XML} mb=${BIN}


.PHONY: run
run: run_py


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
	@echo $(YELLOW)"  make run or run_py"$(NOCOLOR)"- runs the application";
	@echo $(YELLOW)"  make help "$(NOCOLOR)"- shows this message";
	@echo $(YELLOW)"  make all "$(NOCOLOR)"- makes everything needed to run but doesn't run";
	@echo $(YELLOW)"  make data "$(NOCOLOR)"- downloads data as needed";
	@echo $(YELLOW)"  make deps "$(NOCOLOR)"- makes/prepares dependencies";
	@echo $(YELLOW)"  make install-reqs "$(NOCOLOR)"- Installs requirements needed to run this sample on your system.";
	@echo $(YELLOW)"  make uninstall-reqs "$(NOCOLOR)"- Uninstalls requirements that were installed by the sample program.";
	@echo $(YELLOW)"  make default_model "$(NOCOLOR)"- compiles a default model to use when running";
	@echo $(YELLOW)"  make clean "$(NOCOLOR)"- removes all created content";
	@echo ""
	

.PHONY: clean
clean: 
	@echo $(YELLOW)'\n'${APP_NAME}": Cleaning up files...""$(NOCOLOR)"
	rm -f ${CLASSIFIER_GRAPH_BASE_FILENAME}.*



