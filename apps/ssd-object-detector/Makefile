
YELLOW='\033[1;33m'
NOCOLOR='\033[0m'

.PHONY: all
all: run

.PHONY: mobilenet-ssd
mobilenet-ssd:
	@echo $(YELLOW)"\nmaking MobileNet SSD"$(NOCOLOR);
	(cd ../../caffe/SSD_MobileNet; test -f graph || make compile;)

.PHONY: run
run: mobilenet-ssd
	@echo $(YELLOW)"\nRunning ssd-object-detector.py"$(NOCOLOR);
	python3 ssd-object-detector.py

.PHONY: help
help:
	@echo $(YELLOW)"possible make targets: "$(NOCOLOR);
	@echo "  make help - Shows this message";
	@echo "  make - Builds all dependencies, but does not run this program";
	@echo "  make run - Runs this program";

