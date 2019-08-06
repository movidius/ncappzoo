GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
NOCOLOR = '\033[0m'

.PHONY: all
all: deps data

FACE_RELATIVE_DIR = ../../caffe/face-detection-retail-0004
FACE_MODEL_ZOO_DIR = Retail/object_detection/face/sqnet1.0modif-ssd/0004/dldt/FP16
FACE_FILE_NAME_BASE = face-detection-retail-0004

AGE_GEN_RELATIVE_DIR = ../../caffe/AgeGenderNet
AGE_GEN_MODEL_ZOO_DIR = Retail/object_attributes/age_gender/dldt/FP16
AGE_GEN_FILE_NAME_BASE = age-gender-recognition-retail-0013


.PHONY: data
deps: 
	@echo $(YELLOW)"\ngender_age: Downloading the required data..."$(NOCOLOR)


.PHONY: deps
deps: get_ir
	@echo $(YELLOW)"\ngender_age: Making dependencies..."$(NOCOLOR)
	

.PHONY: get_ir
get_ir:	
	@echo $(YELLOW)"\ngender_age: Downloading IRs..."$(NOCOLOR);
	@if [ -e ${AGE_GEN_MODEL_FILE_NAME_BASE}.xml ] && [ -e ${AGE_GEN_MODEL_FILE_NAME_BASE}.bin ] ;\
	then \
		echo " - Age and Gender IRs already exist in the project folder.";\
	else \
		echo " - Age/Gender IR files do not exist in project directory.";\
		echo " - Making Age/Gender IRs...";\
		(cd ${AGE_GEN_RELATIVE_DIR}; make get_ir;);\
		echo " - Copying Age/Gender IR files to current folder..." ;\
		cp ${AGE_GEN_RELATIVE_DIR}/${AGE_GEN_FILE_NAME_BASE}.xml . ;\
		cp ${AGE_GEN_RELATIVE_DIR}/${AGE_GEN_FILE_NAME_BASE}.bin . ;\
	fi;\
	
	@if [ -e ${FACE_GEN_MODEL_FILE_NAME_BASE}.xml ] && [ -e ${FACE_GEN_MODEL_FILE_NAME_BASE}.bin ] ;\
	then \
		echo " - Face detection model IR already exist in the project folder.";\
	else \
		echo " - Face detection model IR files do not exist in project directory.";\
		echo " - Making Face detection IRs...";\
		(cd ${FACE_RELATIVE_DIR}; make get_ir;);\
		echo " - Copying Face IR files to current folder..." ;\
		cp ${FACE_RELATIVE_DIR}/${FACE_FILE_NAME_BASE}.xml . ;\
		cp ${FACE_RELATIVE_DIR}/${FACE_FILE_NAME_BASE}.bin . ;\
	fi


.PHONY:	compile_cpp
compile_cpp:  
	@echo $(YELLOW)"\ngender_age: Building the sample..."$(NOCOLOR)
	(mkdir -p build; cd build; cmake ..; make;)


.PHONY: run
run: run_cpp
	

.PHONY: run_cpp
run_cpp: install-reqs deps data compile_cpp
	@echo $(YELLOW)"\ngender_age: Running the sample..."$(NOCOLOR)
	(cd build; ./gender_age; cd ..;)


.PHONY: install-reqs
install-reqs: 
	@echo $(YELLOW)"\ngender_age: Checking application requirements...\n"$(NOCOLOR)
	@echo "No requirements needed."
	
	
.PHONY: help
help:
	@echo $(YELLOW)"\nPossible make targets: "$(NOCOLOR);
	@echo $(YELLOW)"  make run "$(NOCOLOR)"- runs the gender_age executable example program";
	@echo $(YELLOW)"  make help "$(NOCOLOR)"- shows this message";
	@echo $(YELLOW)"  make all "$(NOCOLOR)"- makes the following: deps, compile_cpp, run";
	@echo $(YELLOW)"  make agenet "$(NOCOLOR)"- compile agenet xml/bin files";
	@echo $(YELLOW)"  make gendernet "$(NOCOLOR)"- compile gendernet xml/bin files";
	@echo $(YELLOW)"  make clean "$(NOCOLOR)"- removes all created content and temporary files";


clean: clean
	@echo $(YELLOW)"\ngender_age: Cleaning up files..."$(NOCOLOR);
	rm -rf build
	rm -f ${AGE_GEN_FILE_NAME_BASE}.xml;
	rm -f ${AGE_GEN_FILE_NAME_BASE}.bin;
	rm -f ${FACE_FILE_NAME_BASE}.xml;
	rm -f ${FACE_FILE_NAME_BASE}.bin;
