.PHONY: all
all: googlenet alexnet squeezenet

.PHONY: run
run: googlenet alexnet squeezenet
	@echo "\nmaking run"
	python3 classifier-gui.py

.PHONY: install-reqs
install-reqs:
	@echo "\nmaking install-reqs"
	sudo apt-get install -y python3-pil.imagetk


.PHONY: googlenet
googlenet: 
	@echo "\nmaking googlenet"
	(cd ../../caffe/GoogLeNet; make compile;)

.PHONY: alexnet
alexnet: 
	@echo "\nmaking alexnet"
	(cd ../../caffe/AlexNet; make compile;)


.PHONY: squeezenet
squeezenet: 
	@echo "\nmaking squeezenet"
	(cd ../../caffe/SqueezeNet; make compile;)

.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make install-reqs - Installs these required packages:";
	@echo "                      python3-pil.imagetk";
	@echo "  make all - everything required to run but doesn't run";
	@echo "  make run - runs the stream_infer program";
	@echo "  make clean - removes all created content and temporary files";

.PHONY: clean
clean:
	@echo "\nmaking clean";
