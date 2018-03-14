.PHONY: all
all: googlenet

.PHONY: run
run: googlenet
	python3 classifier-gui.py

.PHONY: googlenet
googlenet: 
	@echo "\nmaking googlenet"
	(cd ../../caffe/GoogLeNet; make compile;)

.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - makes the following: run";
	@echo "  make run - runs the stream_infer program";
	@echo "  make clean - removes all created content and temporary files";

clean: clean
	@echo "\nmaking clean";
