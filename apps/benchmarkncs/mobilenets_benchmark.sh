#! /bin/bash

for DEPTH in 0.25 0.50 0.75 1.0
do 

    for IMGSIZE in 128 160 192 224
    do
	echo "Calculating FPS for MobileNets($DEPTH $IMGSIZE)"
	(cd ../../tensorflow/mobilenets; make clean; make compile IMGCLASS=$DEPTH IMGSIZE=$IMGSIZE)
	./benchmarkncs.py ../../tensorflow/mobilenets ../../data/images $IMGSIZE $IMGSIZE
    done
done

