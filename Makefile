
GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
NOCOLOR = '\033[0m'
OPEN_MODEL_ZOO = omz

TOPTARGETS := all clean compile_model

SUBDIRS := $(wildcard */.)
INSTALL_PARAM := "yes"
EXIT_PARAM := "no"

$(TOPTARGETS): $(SUBDIRS)
$(SUBDIRS):
	@if [ "$(MAKECMDGOALS)" != "clean" ] || [ "$(MAKECMDGOALS)" = "all" ] || [ -z $(MAKECMDGOALS) ]; \
	then \
		$(MAKE) -C ${OPEN_MODEL_ZOO} INSTALL=${INSTALL_PARAM} EXIT_ON_REQ_NOT_MET=${EXIT_PARAM}; \
	fi; \
	$(eval INSTALL_PARAM = "no")
	@$(MAKE) -C $@ $(MAKECMDGOALS) INSTALL=${INSTALL_PARAM} EXIT_ON_REQ_NOT_MET=${EXIT_PARAM}; \
	echo $(GREEN)"\nAppZoo: "$(YELLOW)"Finished: making "$@ $(MAKECMDGOALS)"\n"$(NOCOLOR)

.PHONY: $(TOPTARGETS) $(SUBDIRS)

.PHONY: help
help:
	@echo "\nPossible Make targets"
	@echo $(YELLOW)"  make help "$(NOCOLOR)"- shows this message"
	@echo $(YELLOW)"  make all "$(NOCOLOR)"- Makes all targets"
	@echo $(YELLOW)"  make clean "$(NOCOLOR)"- Removes all temp files from all directories"
	@echo $(YELLOW)"  make compile_model "$(NOCOLOR)"- Runs compile on all caffe/tensorflow models"

