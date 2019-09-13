GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
NOCOLOR = '\033[0m'

TOPTARGETS := all clean deps data

SUBDIRS := $(wildcard */.)

$(TOPTARGETS): $(SUBDIRS)
$(SUBDIRS):
	$(MAKE) -C $@ $(MAKECMDGOALS)
	@echo $(GREEN)"\nApps: "$(YELLOW)"Finished: making "$@ $(MAKECMDGOALS)$(NOCOLOR)

.PHONY: $(TOPTARGETS) $(SUBDIRS)

.PHONY: help
help:
	@echo "\nPossible make targets: ";
	@echo $(YELLOW)"  make help "$(NOCOLOR)"- shows this message";
	@echo $(YELLOW)"  make data "$(NOCOLOR)"- does make data in each sub directory to download required data";
	@echo $(YELLOW)"  make deps "$(NOCOLOR)"- does make deps in each sub directory to prepare all app dependencies prior to running";
	@echo $(YELLOW)"  make all "$(NOCOLOR)"- does make all in each sub directory";
	@echo $(YELLOW)"  make clean "$(NOCOLOR)"- does make clean in each subdirectory";

