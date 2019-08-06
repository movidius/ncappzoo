GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
NOCOLOR = '\033[0m'

TOPTARGETS := all clean compile_model

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
	@echo $(YELLOW)"  make all "$(NOCOLOR)"- does make all in each sub directory";
	@echo $(YELLOW)"  make compile_model "$(NOCOLOR)"- Compiles models to IR"
	@echo $(YELLOW)"  make clean "$(NOCOLOR)"- does make clean in each subdirectory";

