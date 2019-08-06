GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
NOCOLOR = '\033[0m'


# Filename for the graph file
MNIST_BIN_FILENAME = mnist_inference.bin
MNIST_XML_FILENAME = mnist_inference.xml

MODEL_RELATIVE_DIR = ../../tensorflow/mnist
PROJECT_RELATIVE_DIR = ../../apps/mnist_calc

.PHONY: all
all: deps data

.PHONY: data
data:
	@echo $(YELLOW)"\nmnist_calc: Making data - no data needed."$(NOCOLOR)

.PHONY: deps
deps:
	@echo $(YELLOW)"\nmnist_calc: Making dependencies..."$(NOCOLOR)
	(cd ${MODEL_RELATIVE_DIR}; make compile_model; cd ${PROJECT_RELATIVE_DIR}; cp ${MODEL_RELATIVE_DIR}/${MNIST_BIN_FILENAME} ./${MNIST_BIN_FILENAME}; cp ${MODEL_RELATIVE_DIR}/${MNIST_XML_FILENAME} ./${MNIST_XML_FILENAME};)


.PHONY: install-reqs
install-reqs: 
	@echo $(YELLOW)"\nmnist_calc: Checking application requirements...\n"$(NOCOLOR)
	@echo "No requirements needed."
	
	
.PHONY: run
run: install-reqs deps data
	@echo $(YELLOW)"\nmnist_calc: Running the sample..."$(NOCOLOR)
	python3 ./touchcalc.py


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
	@echo $(YELLOW)"\nmnist_calc: Cleaning up files..."$(NOCOLOR)
	rm -f ${MNIST_BIN_FILENAME}
	rm -f ${MNIST_XML_FILENAME}


