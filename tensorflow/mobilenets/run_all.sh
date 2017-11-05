#! /bin/bash

for CLASS in 0.25 0.50 0.75 1.0
do 
    for SIZE in 128 160 192 224
    do
	echo "Running $CLASS $SIZE"
	make clean run IMGCLASS=$CLASS IMGSIZE=$SIZE
    done
done

