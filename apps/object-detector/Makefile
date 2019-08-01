
YELLOW='\033[1;33m'
NOCOLOR='\033[0m'

.PHONY: all
all: mobilenet-ssd tinyyolo

.PHONY: mobilenet-ssd
mobilenet-ssd:
	@echo $(YELLOW)"\nmaking MobileNet SSD"$(NOCOLOR);
	(cd ../../caffe/SSD_MobileNet; test -f graph || make compile;)

.PHONY: tinyyolo
tinyyolo:
	@echo $(YELLOW)"\nmaking Tiny Yolo"$(NOCOLOR);
	(cd ../../caffe/TinyYolo; test -f graph || make compile;)

.PHONY: run_ssd
run_ssd: mobilenet-ssd
	@echo $(YELLOW)"\nRunning object-detector.py for MobileNet SSD"$(NOCOLOR);
	python3 object-detector.py

.PHONY: run_tinyyolo
run_tinyyolo: tinyyolo
	@echo $(YELLOW)"\nRunning object-detector.py for Tiny Yolo"$(NOCOLOR);
	python3 object-detector.py -n TinyYolo -g ../../caffe/TinyYolo/graph -l ../../caffe/TinyYolo/labels.txt -M 0 0 0 -S 0.00392 -D 448 448 -c rgb

.PHONY: run
run: run_ssd run_tinyyolo

.PHONY: help
help:
	@echo $(YELLOW)"possible make targets: "$(NOCOLOR);
	@echo "  make help - Shows this message";
	@echo "  make - Builds all dependencies, but does not run this program";
	@echo "  make run_ssd - Runs Object detector on Mobilenet SSD";
	@echo "  make run_tinyyolo - Runs Object detector on Tiny Yolo";
	@echo "  make run - Runs Object detector on both Mobilenet SSD and Tiny Yolo";


.PHONY: clean
clean:
	@echo "\nmaking clean";
	@echo "  Nothing to clean";

