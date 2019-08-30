
GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
NOCOLOR = '\033[0m'

TOPTARGETS := clean

SUBDIRS := ilsvrc12

$(TOPTARGETS): $(SUBDIRS)
$(SUBDIRS):
	$(MAKE) -C $@ $(MAKECMDGOALS)
	@echo $(GREEN)"\nData: "$(YELLOW)"Finished: making "$@ $(MAKECMDGOALS)$(NOCOLOR)

.PHONY: $(TOPTARGETS) $(SUBDIRS)

all compile_model run:

.PHONY: help
help:
	@echo "\nPossible make targets: ";
	@echo $(YELLOW)"  make help "$(NOCOLOR)"- shows this message"
	@echo $(YELLOW)"  make clean "$(NOCOLOR)"- Removes all temp files from all directories"
