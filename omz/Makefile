NCCOMPILE = mo.py

GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
RED = '\033[1;31m'
NOCOLOR = '\033[0m'

TOPTARGETS := all clean compile_model
MODEL_ZOO_DIR = open_model_zoo
INSTALL = "yes"
EXIT_ON_REQ_NOT_MET = "yes"
CHECK_PYAML = python3 -c "import yaml"


.PHONY: all
all: install-reqs deps data


.PHONY: data
data:
	@echo $(YELLOW)"\nopen_model_zoo: Downloading required data..."$(NOCOLOR)
	@if [ -d ${MODEL_ZOO_DIR} ] ; \
	then \
		echo "Open model zoo already exists. No need to clone.";\
	else \
		echo "Open model zoo does not exist. Cloning model zoo...";\
		git clone https://github.com/opencv/open_model_zoo --branch 2019_R2; \
	fi
	
	
.PHONY: install-reqs
install-reqs:
	@echo $(YELLOW)"\nopen_model_zoo: Checking requirements..."$(NOCOLOR)
	@echo "Checking pyyaml..."
	@if ${shell} ${CHECK_PYAML} 2> /dev/null; \
	then \
	    echo " - pyyaml already installed.\n" ; \
	else \
		echo $(YELLOW)"\nATTENTION! "$(NOCOLOR)"pyyaml is required to run the model downloader from the model zoo."; \
		echo "Use the command: " $(YELLOW)"pip3 install pyyaml --user"$(NOCOLOR)" to install requirement"; \
		if [ "$(INSTALL)" = "yes" ] ;\
		then \
			read -p " or type yes to install the requirement: " user_reply; \
			if [ $$user_reply = "yes" ] || [ $$user_reply = "Y" ] || [ $$user_reply = "y" ]; \
			then \
				echo " - Installing pyyaml..."; \
				pip3 install pyyaml --user; \
			else \
				echo $(RED)" - Not installing pyyaml..."; \
				echo " - Models for samples that require the model downloader will not be downloaded."$(NOCOLOR); \
				read -p " Press ENTER key to continue " junk; \
				if [ "$(EXIT_ON_REQ_NOT_MET)" = "yes" ]; \
				then \
					echo $(RED)" *** Requirements not met. Exiting... ***"$(NOCOLOR) ; \
					exit 2; \
				fi; \
			fi; \
		fi; \
	fi
	
	
.PHONY: deps
deps: 
	@echo $(YELLOW)"\nopen_model_zoo: Making dependencies..."$(NOCOLOR)
	

.PHONY: help
help:
	@echo "\nPossible Make targets"
	@echo $(YELLOW)"  make help "$(NOCOLOR)"- shows this message"
	@echo $(YELLOW)"  make all "$(NOCOLOR)"- Makes all targets"
	@echo $(YELLOW)"  make clean "$(NOCOLOR)"- Removes all temp files from all directories"
	@echo $(YELLOW)"  make compile_model "$(NOCOLOR)"- Compiles models to IR"
	
	
.PHONY: clean
clean:
	@echo $(YELLOW)"\nopen_model_zoo: Cleaning up files..."$(NOCOLOR)
	rm -rf open_model_zoo
