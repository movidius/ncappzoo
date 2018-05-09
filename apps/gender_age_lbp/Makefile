


.PHONY: all
all: gender_age_lbp

.PHONY: gendernet
gendernet: 
	@echo "\nmaking gendernet"
	(cd ../../caffe/GenderNet; make compile; cd ../../apps/gender_age_lbp; cp ../../caffe/GenderNet/graph ./gendernet.graph;)

.PHONY: agenet
agenet: 
	@echo "\nmaking agenet"
	(cd ../../caffe/AgeNet; make compile; cd ../../apps/gender_age_lbp; cp ../../caffe/AgeNet/graph ./agenet.graph;)

.PHONY:	gender_age_lbp
gender_age_lbp: gendernet agenet
	@echo "\nmaking gender_age_lbp"
	mkdir -p gender_graph;
	mkdir -p age_graph;
	wget -nc https://raw.githubusercontent.com/opencv/opencv/master/data/lbpcascades/lbpcascade_frontalface_improved.xml;
	cp gendernet.graph gender_graph/graph;
	cp agenet.graph age_graph/graph;
	./compile.sh;
	@echo "Created cpp/gender_age_lbp executable"

.PHONY: run
run: gender_age_lbp
	@echo "\nmaking run";
	cd cpp; ./gender_age_lbp; cd ..;

.PHONY: run_gender
run_gender: gender_age_lbp
	@echo "\nmaking run gender";
	cd cpp; ./gender_age_lbp gender; cd ..;

.PHONY: run_age
run_age: gender_age_lbp
	@echo "\nmaking run age";
	cd cpp; ./gender_age_lbp age; cd ..;

.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - makes the following: gender_age_lbp, run";
	@echo "  make gender_age_lbp - builds the gender_age_lbp executable example";
	@echo "  make run - runs the gender_age_lbp executable example program";
	@echo "  make clean - removes all created content and temporary files";

clean: clean
	@echo "\nmaking clean";
	rm -f cpp/gender_age_lbp;
	rm -f lbpcascade_frontalface_improved.xml
	rm -rf gender_graph;
	rm -rf age_graph;
	rm -f gendernet.graph;
	rm -f agenet.graph;

