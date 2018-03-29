


.PHONY: all
all: tf_inception_v1 tf_inception_v3

.PHONY: tf_inception_v1
tf_inception_v1: 
	@echo "\nmaking tf inception_v1"
	(cd ../../tensorflow/inception; make compile VERSION=v1;)

.PHONY: tf_inception_v3
tf_inception_v3: 
	@echo "\nmaking tf inception_v3"
	(cd ../../tensorflow/inception; make compile VERSION=v3;)

.PHONY: run
run: tf_inception_v1
	@echo "\nmaking run";
	python3 MultiStick_Inception.py 

.PHONY: run_inception_v1
run_inception_v1: tf_inception_v1
	@echo "\nmaking run_inception_v1";
	python3 MultiStick_Inception.py ../../tensorflow/inception/model/v1/ ../../data/images/ 224

.PHONY: run_inception_v3
run_inception_v3: tf_inception_v3
	@echo "\nmaking run_inception_v3";
	python3 MultiStick_Inception.py ../../tensorflow/inception/model/v3/ ../../data/images/ 229

.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - makes the following: run";
	@echo "  make run - runs the program and prompts for network and images dirs";
	@echo "  make run_inception_v1 - runs the program with inception_v1";
	@echo "  make run_inception_v3 - runs the program with inception_v3";



.PHONY: clean
clean:
	@echo "Cleaning: ";

