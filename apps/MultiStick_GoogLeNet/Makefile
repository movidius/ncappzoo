


.PHONY: all
all: googlenet

.PHONY: googlenet
googlenet: 
	@echo "\nmaking googlenet"
	(cd ../../caffe/GoogLeNet; make compile;)

.PHONY: python3-tk
python3-tk: 
	@echo "\nChecking python3-tk dependency"
	./python3-tk_dep_check.sh

.PHONY: run
run: googlenet python3-tk
	@echo "\nmaking run";
	python3 MultiStick_GoogLeNet.py

.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - makes the following: run";
	@echo "  make run - runs the multistick_googlenet program";
	@echo "  make clean - removes all created content and temporary files";

clean: clean
	@echo "\nmaking clean";


