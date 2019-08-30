GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
NOCOLOR = '\033[0m'

# filenames for the graph files that we'll copy to this directory.
FACE_RELATIVE_DIR = ../../caffe/face-detection-retail-0004
FACE_MODEL_ZOO_DIR = Retail/object_detection/face/sqnet1.0modif-ssd/0004/dldt/FP16
FACE_FILE_NAME_BASE = face-detection-retail-0004

AGE_GEN_RELATIVE_DIR = ../../caffe/AgeGenderNet
AGE_GEN_MODEL_ZOO_DIR = Retail/object_attributes/age_gender/dldt/FP16
AGE_GEN_FILE_NAME_BASE = age-gender-recognition-retail-0013


# name of the directory and the base name of the main python file (minus the .py extension)
APP_NAME = gender_age

.PHONY: all
all: deps data compile_cpp


.PHONY: data
data: 
	@echo $(YELLOW)'\n'${APP_NAME}": No data required."$(NOCOLOR)


.PHONY: deps
deps: get_ir
	@echo $(YELLOW)'\n'${APP_NAME}": Making dependencies..."$(NOCOLOR)
	


.PHONY: default_model
default_model: get_ir
	@echo $(YELLOW)'\n'${APP_NAME}": Making default models..."$(NOCOLOR)


.PHONY: get_ir
get_ir:	
	@echo $(YELLOW)'\n'${APP_NAME}": Downloading IRs..."$(NOCOLOR);
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
	@echo $(YELLOW)'\n'${APP_NAME}": Building the sample..."$(NOCOLOR)
	(mkdir -p build; cd build; cmake ..; make;)


.PHONY: run
run: run_cpp
	

.PHONY: run_cpp
run_cpp: deps data compile_cpp
	@echo $(YELLOW)'\n'${APP_NAME}": Running the sample..."$(NOCOLOR)
	(cd build; ./${APP_NAME}; cd ..;)


.PHONY: install-reqs
install-reqs: 
	@echo $(YELLOW)"\n"$(APP_NAME)": Checking installation requirements..."$(NOCOLOR)
	@echo "No requirements needed."	


.PHONY: uninstall-reqs
uninstall-reqs: 
	@echo $(YELLOW)'\n'${APP_NAME}": Uninstalling requirements..."$(NOCOLOR)
	@echo "Nothing to uninstall."
	

.PHONY: help
help:
	@echo "\nPossible make targets: ";
	@echo $(YELLOW)"  make run or run_cpp"$(NOCOLOR)"- runs the application";
	@echo $(YELLOW)"  make help "$(NOCOLOR)"- shows this message";
	@echo $(YELLOW)"  make all "$(NOCOLOR)"- makes everything needed to run but doesn't run";
	@echo $(YELLOW)"  make data "$(NOCOLOR)"- downloads data as needed";
	@echo $(YELLOW)"  make deps "$(NOCOLOR)"- makes/prepares dependencies";
	@echo $(YELLOW)"  make install-reqs "$(NOCOLOR)"- Installs requirements needed to run this sample on your system.";
	@echo $(YELLOW)"  make uninstall-reqs "$(NOCOLOR)"- Uninstalls requirements that were installed by the sample program.";
	@echo $(YELLOW)"  make default_model "$(NOCOLOR)"- compiles a default model to use when running";
	@echo $(YELLOW)"  make get_ir "$(NOCOLOR)"- gets the age-gender/face IR files from the model zoo";
	@echo $(YELLOW)"  make clean "$(NOCOLOR)"- removes all created content"
	@echo ""


clean: 
	@echo $(YELLOW)'\n'${APP_NAME}": Cleaning up files..."$(NOCOLOR);
	rm -rf build
	rm -f ${AGE_GEN_FILE_NAME_BASE}.xml;
	rm -f ${AGE_GEN_FILE_NAME_BASE}.bin;
	rm -f ${FACE_FILE_NAME_BASE}.xml;
	rm -f ${FACE_FILE_NAME_BASE}.bin;
	
