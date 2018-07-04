
YELLOW='\033[1;33m'
NOCOLOR='\033[0m'
VIDEO?=0

.PHONY: all
all: run

.PHONY: mobilenet-ssd
mobilenet: 
	@echo $(YELLOW)"\nmaking MobileNet SSD"$(NOCOLOR);
	(cd ../../caffe/SSD_MobileNet; test -f graph || make compile;)

.PHONY: run
run: mobilenet-ssd
	@echo $(YELLOW)"\nRunning live-object-detector.py"$(NOCOLOR);
	python3 live-object-detector.py --video $(VIDEO)

.PHONY: help
help:
	@echo $(YELLOW)"possible make targets: "$(NOCOLOR);
	@echo "  make help - Shows this message";
	@echo "  make - Builds all dependencies, but does not run this program";
	@echo "  make run - Runs this program";

.PHONY: clean
clean:
	@echo "\nmaking clean";
	@echo "  Nothing to clean";
