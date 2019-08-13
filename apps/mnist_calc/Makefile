
GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
NOCOLOR = '\033[0m'


# Filename for the graph file
MNIST_BIN_FILENAME = mnist_inference.bin
MNIST_XML_FILENAME = mnist_inference.xml

MODEL_RELATIVE_DIR = ../../tensorflow/mnist
PROJECT_RELATIVE_DIR = ../../apps/mnist_calc

# name of the directory and the base name of the main python file (minus the .py extension)
APP_NAME = mnist_calc

.PHONY: all
all: deps data

.PHONY: data
data:
	@echo $(YELLOW)'\n'${APP_NAME}": No data needed."$(NOCOLOR)

.PHONY: deps
deps: default_model
	@echo $(YELLOW)'\n'${APP_NAME}": Making dependencies..."$(NOCOLOR)


.PHONY: compile_default_model
compile_default_model:
	@echo $(YELLOW)'\n'${APP_NAME}": Compiling default model to IR..."$(NOCOLOR)
	(cd ${MODEL_RELATIVE_DIR}; make compile_model;); \
	echo " - Copying IRs to project directory..."; \
	(cp ${MODEL_RELATIVE_DIR}/${MNIST_BIN_FILENAME} ./${MNIST_BIN_FILENAME}; cp ${MODEL_RELATIVE_DIR}/${MNIST_XML_FILENAME} ./${MNIST_XML_FILENAME};) 
	
	
.PHONY: default_model
default_model: compile_default_model
	@echo $(YELLOW)'\n'${APP_NAME}": Making default models..."$(NOCOLOR)


.PHONY: install-reqs
install-reqs: 
	@echo $(YELLOW)"\n"$(APP_NAME)": Checking installation requirements..."$(NOCOLOR)
	@echo "No requirements needed."	
	
.PHONY: run
run: install-reqs deps data
	@echo $(YELLOW)'\n'${APP_NAME}": Running the sample..."$(NOCOLOR)
	python3 ./$(APP_NAME).py


.PHONY: help
help:
	@echo "\nPossible make targets: ";
	@echo $(YELLOW)"  make run "$(NOCOLOR)"- runs the application";
	@echo $(YELLOW)"  make help "$(NOCOLOR)"- shows this message";
	@echo $(YELLOW)"  make all "$(NOCOLOR)"- makes everything needed to run but doesn't run";
	@echo $(YELLOW)"  make data "$(NOCOLOR)"- downloads data as needed";
	@echo $(YELLOW)"  make deps "$(NOCOLOR)"- makes/prepares dependencies";
	@echo $(YELLOW)"  make install-reqs "$(NOCOLOR)"- Installs requirements needed to run this sample on your system.";
	@echo $(YELLOW)"  make default_model "$(NOCOLOR)"- compiles a default model to use when running";
	@echo $(YELLOW)"  make clean "$(NOCOLOR)"- removes all created content";
	@echo ""


.PHONY: clean
clean: 
	@echo $(YELLOW)'\n'${APP_NAME}": Cleaning up files..."$(NOCOLOR)
	rm -f ${MNIST_BIN_FILENAME}
	rm -f ${MNIST_XML_FILENAME}


