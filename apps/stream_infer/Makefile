
.PHONY: all
all: squeezenet

.PHONY: gendernet
gendernet: 
	@echo "\nmaking gendernet"
	(cd ../../caffe/GenderNet; make compile;)

.PHONY: googlenet
googlenet: 
	@echo "\nmaking googlenet"
	(cd ../../caffe/GoogLeNet; make compile;)

.PHONY: squeezenet
squeezenet: 
	@echo "\nmaking squeezenet"
	(cd ../../caffe/SqueezeNet; make compile;)

.PHONY: alexnet
alexnet:
	@echo "\nmaking alexnet"
	(cd ../../caffe/AlexNet; make compile;)

.PHONY: run
run: squeezenet
	@echo "\nmaking run";
	python3 stream_infer.py

.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - makes everything needed to prepare for running";
	@echo "  make run - runs the stream_infer program";
	@echo "  make gendernet - compiles the GenderNet network";
	@echo "  make googlenet - compiles the GoogLeNet network";
	@echo "  make squeezenet - compiles the SqueezeNet network";
	@echo "  make alexnet - compiles the AlexNet network";
	@echo "  make clean - removes all created content and temporary files";

clean: clean
	@echo "\nmaking clean";


