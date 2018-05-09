


.PHONY: all
all: hello_ncs_cpp


.PHONY:	hello_ncs_cpp
hello_ncs_cpp: 
	@echo "\nmaking hello_ncs_cpp"
	g++ cpp/hello_ncs.cpp -o cpp/hello_ncs_cpp -lmvnc
	@echo "Created cpp/hello_ncs_cpp executable"

.PHONY: run
run: hello_ncs_cpp
	@echo "\nmaking run"
	cd cpp; ./hello_ncs_cpp; cd ..

.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - makes the following: cpp, run_cpp";
	@echo "  make hello_ncs_cpp - builds the hello_ncs_cpp executable example";
	@echo "  make run - runs the hello_ncs_cpp executable example program";
	@echo "  make clean - removes all created content and temporary files";

clean: clean
	@echo "\nmaking clean";
	rm -f cpp/hello_ncs_cpp;
