GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
NOCOLOR = '\033[0m'

NCCOMPILE = mvNCCompile
NCPROFILE = mvNCProfile
NCCHECK   = mvNCCheck
APP_NAME = birds

# filenames for the graph files that we'll copy to this directory.
YOLO_TINY_IR = tiny-yolo-v1_53000
GOOGLENET_IR = googlenet-v1
DOG_FILENAME = dog.jpg
IMAGE_DIR = ./images

GET_IMAGES = mkdir -p ${IMAGE_DIR}; \
             wget -c --no-cache -P ${IMAGE_DIR} https://www.nps.gov/sajh/learn/nature/images/Eagles-2_Rebecca.jpg; \
             wget -c --no-cache -P ${IMAGE_DIR} https://raw.githubusercontent.com/nealvis/bird_images/master/jas_birds_10.jpg; \
             wget -c --no-cache -P ${IMAGE_DIR} https://raw.githubusercontent.com/nealvis/bird_images/master/jas_birds_2.jpg; \
             wget -c --no-cache -P ${IMAGE_DIR} https://raw.githubusercontent.com/nealvis/bird_images/master/jas_birds_8.jpg; \
             wget -c --no-cache -P ${IMAGE_DIR} https://raw.githubusercontent.com/nealvis/bird_images/master/nps_birds_1.jpg; \
             wget -c --no-cache -P ${IMAGE_DIR} https://raw.githubusercontent.com/nealvis/bird_images/master/nps_birds_2.jpg; \
             wget -c --no-cache -P ${IMAGE_DIR} https://raw.githubusercontent.com/nealvis/bird_images/master/nps_birds_3.jpg


.PHONY: all
all: data deps compile_model


.PHONY: deps
deps: compile_model
	@echo $(YELLOW)"\n"$(APP_NAME)": Making the required dependencies... None needed."$(NOCOLOR)
	

.PHONY: data
data: images ilsvrc12
	@echo $(YELLOW)"\n"$(APP_NAME)": Getting the required data..."$(NOCOLOR)
	

.PHONY: images
images:
	@echo $(YELLOW)"\n"$(APP_NAME)": Downloading images..."$(NOCOLOR)
	${GET_IMAGES};


.PHONY: ilsvrc12
ilsvrc12:
	@echo $(YELLOW)"\n"$(APP_NAME)": Downloading the Imagenet data..."$(NOCOLOR)
	(cd ../../data/ilsvrc12; make)


.PHONY: install-reqs
install-reqs:
	@echo $(YELLOW)"\n"$(APP_NAME)": Checking installation requirements...None needed."$(NOCOLOR)


.PHONY: googlenet
googlenet: 
	@echo $(YELLOW)"\n"$(APP_NAME)": Downloading and Compiling GoogLeNet..."$(NOCOLOR)
	(cd ../../caffe/GoogLeNet; make compile_model; cd ../../apps/birds; cp ../../caffe/GoogLeNet/googlenet-v1.xml .; cp ../../caffe/GoogLeNet/googlenet-v1.bin .;) 


.PHONY: tiny_yolo
tiny_yolo: 
	@echo $(YELLOW)"\n"$(APP_NAME)": Downloading and Compiling Tiny Yolo v1..."$(NOCOLOR)
	(cd ../../caffe/TinyYolo; make compile_model; cd ../../apps/birds; cp ../../caffe/TinyYolo/tiny-yolo-v1_53000.xml .; cp ../../caffe/TinyYolo/tiny-yolo-v1_53000.bin .;)


.PHONY: compile_model
compile_model: googlenet tiny_yolo
	@echo $(YELLOW)"\n"$(APP_NAME)": Compiling models to IR..."$(NOCOLOR)
	

.PHONY: run_py
run_py: install-reqs deps data compile_model
	@echo $(YELLOW)"\n"$(APP_NAME)": Running the Python sample..."$(NOCOLOR)
	python3 ./birds.py


.PHONY: run
run: run_py
	@echo ""


.PHONY: help
help:
	@echo "\n Possible make targets: ";
	@echo $(YELLOW)"  make help "$(NOCOLOR)"- shows this message";
	@echo $(YELLOW)"  make all "$(NOCOLOR)"- makes everything needed to run but doesn't run";
	@echo $(YELLOW)"  make compile_model "$(NOCOLOR)"- compiles required networks with Model Optimizer tool to create graph files";
	@echo $(YELLOW)"  make run or run_py "$(NOCOLOR)"- runs the birds.py python example program";
	@echo $(YELLOW)"  make clean "$(NOCOLOR)"- removes all created content in this directory.";
	@echo ""
    
    
.PHONY: clean
clean: 
	@echo $(YELLOW)"\n"$(APP_NAME)": Cleaning up files..."$(NOCOLOR)
	rm -f ${YOLO_TINY_IR}.xml
	rm -f ${GOOGLENET_IR}.xml
	rm -f ${YOLO_TINY_IR}.bin
	rm -f ${GOOGLENET_IR}.bin
	rm -rf ${IMAGE_DIR}


