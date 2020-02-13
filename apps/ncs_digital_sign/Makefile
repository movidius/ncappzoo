GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
NOCOLOR = '\033[0m'

.PHONY: all
all: deps data compile_cpp

FACE_RELATIVE_DIR = ../../networks/face_detection_retail_0004
FACE_FILE_NAME_BASE = face-detection-retail-0004

AGE_GEN_RELATIVE_DIR = ../../networks/age_gender_net
AGE_GEN_FILE_NAME_BASE = age-gender-recognition-retail-0013


.PHONY: data
data: 
	@echo $(YELLOW)"\nncs_digital_sign: No data download required..."$(NOCOLOR)


.PHONY: deps
deps: get_ir
	@echo $(YELLOW)"\nncs_digital_sign: Making dependencies..."$(NOCOLOR)
	

.PHONY: get_ir
get_ir:	
	@echo $(YELLOW)"\nncs_digital_sign: Downloading IRs..."$(NOCOLOR);
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
	@echo $(YELLOW)"\nncs_digital_sign: Building the sample..."$(NOCOLOR)
	@if [ -z "$(INTEL_OPENVINO_DIR)" ] ; \
	then \
		echo "Please initiate the Intel OpenVINO environment by going to the installation directory for openvino and running the setupvars.sh file in the bin folder." ; \
		exit 1 ; \
	else \
		echo "Intel OpenVINO environment is already set!" ; \
	fi
	(mkdir -p build; cd build; cmake ..; make;)


.PHONY: run
run: run_cpp


.PHONY: compile_model
compile_model:
	@echo $(YELLOW)"\nncs_digital_sign: Not applicable..."$(NOCOLOR)
	
	
.PHONY: run_cpp
run_cpp: install-reqs deps data compile_cpp
	@echo $(YELLOW)"\nncs_digital_sign: Running the sample..."$(NOCOLOR)
	(cd build; ./ncs_digital_sign; cd ..;)


.PHONY: install-reqs
install-reqs: 
	@echo $(YELLOW)"\nncs_digital_sign: Checking application requirements...\n"$(NOCOLOR)
	@echo "No requirements needed."

.PHONY: uninstall-reqs
uninstall-reqs: 
	@echo $(YELLOW)"\nncs_digital_sign: Checking application requirements for uninstall...\n"$(NOCOLOR)
	@echo "No requirements needed to be uninstalled."
	
.PHONY: help
help:
	@echo $(YELLOW)"\nPossible make targets: "$(NOCOLOR);
	@echo $(YELLOW)"  make run "$(NOCOLOR)"- runs the gender_age executable example program";
	@echo $(YELLOW)"  make help "$(NOCOLOR)"- shows this message";
	@echo $(YELLOW)"  make all "$(NOCOLOR)"- makes the following: deps, compile_cpp, run";
	@echo $(YELLOW)"  make deps "$(NOCOLOR)"- download and prepare networks";
	@echo $(YELLOW)"  make install-reqs "$(NOCOLOR)"- Not applicable";
	@echo $(YELLOW)"  make uninstall-reqs "$(NOCOLOR)"- Not applicable";
	@echo $(YELLOW)"  make compile_cpp "$(NOCOLOR)"- Builds application";
	@echo $(YELLOW)"  make compile_model "$(NOCOLOR)"- Not applicable";
	@echo $(YELLOW)"  make clean "$(NOCOLOR)"- removes all created content and temporary files";


clean: 
	@echo $(YELLOW)"\nncs_digital_sign: Cleaning up files..."$(NOCOLOR);
	rm -rf build
	rm -f ${AGE_GEN_FILE_NAME_BASE}.xml;
	rm -f ${AGE_GEN_FILE_NAME_BASE}.bin;
	rm -f ${FACE_FILE_NAME_BASE}.xml;
	rm -f ${FACE_FILE_NAME_BASE}.bin;
