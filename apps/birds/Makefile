GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
NOCOLOR = '\033[0m'


APP_NAME = birds

# filenames for the graph files that we'll copy to this directory.
YOLO_TINY_IR = tiny-yolo-v1_53000
GOOGLENET_IR = googlenet-v1
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
all: data deps


.PHONY: deps
deps: default_model
	@echo $(YELLOW)"\n"$(APP_NAME)": Making the required dependencies..."$(NOCOLOR)
	

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
	@echo $(YELLOW)"\n"$(APP_NAME)": Checking installation requirements..."$(NOCOLOR)
	@echo "No requirements needed."


.PHONY: uninstall-reqs
uninstall-reqs: 
	@echo $(YELLOW)'\n'${APP_NAME}": Uninstalling requirements..."$(NOCOLOR)
	@echo "Nothing to uninstall."
	

.PHONY: googlenet
googlenet: 
	@echo $(YELLOW)"\n"$(APP_NAME)": Downloading and Compiling GoogLeNet..."$(NOCOLOR)
	(cd ../../caffe/GoogLeNet; make compile_model; cd ../../apps/birds; cp ../../caffe/GoogLeNet/googlenet-v1.xml .; cp ../../caffe/GoogLeNet/googlenet-v1.bin .;) 


.PHONY: tiny_yolo
tiny_yolo: 
	@echo $(YELLOW)"\n"$(APP_NAME)": Downloading and Compiling Tiny Yolo v1..."$(NOCOLOR)
	(cd ../../caffe/TinyYolo; make compile_model; cd ../../apps/birds; cp ../../caffe/TinyYolo/tiny-yolo-v1_53000.xml .; cp ../../caffe/TinyYolo/tiny-yolo-v1_53000.bin .;)


.PHONY: default_model
default_model: googlenet tiny_yolo
	@echo $(YELLOW)"\n"$(APP_NAME)": Compiling models to IR..."$(NOCOLOR)
	

.PHONY: run_py
run_py: deps data
	@echo $(YELLOW)"\n"$(APP_NAME)": Running the Python sample..."$(NOCOLOR)
	python3 ./birds.py


.PHONY: run
run: run_py
	@echo ""


.PHONY: help
help:
	@echo "\nPossible make targets: ";
	@echo $(YELLOW)"  make run or run_py"$(NOCOLOR)"- runs the application";
	@echo $(YELLOW)"  make help "$(NOCOLOR)"- shows this message";
	@echo $(YELLOW)"  make all "$(NOCOLOR)"- makes everything needed to run but doesn't run";
	@echo $(YELLOW)"  make data "$(NOCOLOR)"- downloads data as needed";
	@echo $(YELLOW)"  make deps "$(NOCOLOR)"- makes/prepares dependencies";
	@echo $(YELLOW)"  make install-reqs "$(NOCOLOR)"- Installs requirements needed to run this sample on your system.";
	@echo $(YELLOW)"  make uninstall-reqs "$(NOCOLOR)"- Uninstalls requirements that were installed by the sample program.";
	@echo $(YELLOW)"  make default_model "$(NOCOLOR)"- compiles a default model to use when running";
	@echo $(YELLOW)"  make clean "$(NOCOLOR)"- removes all created content";
	@echo ""
    
    
.PHONY: clean
clean: 
	@echo $(YELLOW)"\n"$(APP_NAME)": Cleaning up files..."$(NOCOLOR)
	rm -f ${YOLO_TINY_IR}.xml
	rm -f ${GOOGLENET_IR}.xml
	rm -f ${YOLO_TINY_IR}.bin
	rm -f ${GOOGLENET_IR}.bin
	rm -rf ${IMAGE_DIR}


