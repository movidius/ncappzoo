GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
NOCOLOR = '\033[0m'


CHECK_IMAGETK = python3 -c "from PIL import Image, ImageTk" 
CHECK_SKIMAGE = python3 -c "import skimage" 


.PHONY: all
all: deps data


.PHONY: run
run: run_py
	
	
.PHONY: run_py
run_py: install-reqs deps googlenet alexnet squeezenet 
	@echo $(YELLOW)"\nclassifier_gui: Running the sample..."$(NOCOLOR)
	python3 classifier_gui.py


.PHONY: deps
deps: googlenet alexnet squeezenet
	@echo $(YELLOW)"\nclassifier_gui: Making dependencies..."$(NOCOLOR)


.PHONY: data
deps:
	@echo $(YELLOW)"\nclassifier_gui: Making required data..."$(NOCOLOR)
	(cd ../../data/ilsvrc12; make)


.PHONY: install-reqs
install-reqs: 
	@echo $(YELLOW)"\nclassifier_gui: Checking application requirements...\n"$(NOCOLOR)
	@echo "Checking python3-pil.imagetk..."
	@if $(shell) ${CHECK_IMAGETK} 2> /dev/null; \
	then \
		echo " - python3-pil.imagetk already installed.\n"; \
	else \
		echo $(YELLOW)"ATTENTION! "$(NOCOLOR)"python3-pil.imagetk is required to run this sample.";\
		echo "Use the command:" $(YELLOW)"sudo apt-get install -y python3-pil.imagetk"$(NOCOLOR)" to install requirement.\n"; \
		exit 1;\
	fi; \
	
	@echo "Checking scikit-image..."
	@if ${shell} ${CHECK_SKIMAGE} 2> /dev/null; \
	then \
	    echo " - scikit-image already installed.\n" ; \
	else \
		echo $(YELLOW)"\nATTENTION! "$(NOCOLOR)"scikit-image is required to run this sample."; \
		echo "Use the command: " $(YELLOW)"pip3 install scikit-image --user"$(NOCOLOR)" to install requirement."; \
		exit 1; \
	fi
	
.PHONY: googlenet
googlenet: 
	@echo $(YELLOW)"\nclassifier_gui: Making googlenet..."$(NOCOLOR)
	(cd ../../caffe/GoogLeNet; make compile_model;)


.PHONY: alexnet
alexnet: 
	@echo $(YELLOW)"\nclassifier_gui: Making alexnet..."$(NOCOLOR)
	(cd ../../caffe/AlexNet; make compile_model;)


.PHONY: squeezenet
squeezenet: 
	@echo $(YELLOW)"\nclassifier_gui: Making squeezenet..."$(NOCOLOR)
	(cd ../../caffe/SqueezeNet; make compile_model;)


.PHONY: help
help:
	@echo "\nPossible make targets: ";
	@echo $(YELLOW)"  make run "$(NOCOLOR)"- runs the stream_infer program";
	@echo $(YELLOW)"  make help "$(NOCOLOR)"- shows this message";
	@echo $(YELLOW)"  make install-reqs "$(NOCOLOR)"- Installs these required packages:";
	@echo $(YELLOW)"                      python3-pil.imagetk";
	@echo $(YELLOW)"  make all "$(NOCOLOR)"- everything required to run but doesn't run";
	@echo $(YELLOW)"  make clean "$(NOCOLOR)"- removes all created content and temporary files";

.PHONY: clean_googlenet
clean_googlenet: 
	@echo $(YELLOW)"\nclassifier_gui: Cleaning up googlenet..."$(NOCOLOR)
	(cd ../../caffe/GoogLeNet; make clean;)

.PHONY: clean_alexnet
clean_alexnet: 
	@echo $(YELLOW)"\nclassifier_gui: Cleaning up alexnet..."$(NOCOLOR)
	(cd ../../caffe/AlexNet; make clean;)


.PHONY: clean_squeezenet
clean_squeezenet: 
	@echo $(YELLOW)"\nclassifier_gui: Cleaning up squeezenet..."$(NOCOLOR)
	(cd ../../caffe/SqueezeNet; make clean;)
	
.PHONY: clean
clean: clean_alexnet clean_squeezenet clean_googlenet
	@echo $(YELLOW)"\nclassifier_gui: Cleaning up files..."$(NOCOLOR)
	
