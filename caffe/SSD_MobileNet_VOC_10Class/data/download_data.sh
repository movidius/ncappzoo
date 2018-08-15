#!/bin/bash

#Download VOC dataset and extract the tar ball

if [ -f VOCtrainval_11-May-2012.tar ]
then 
    echo "VOCtrainval_11-May-2012.tar found"
else
    echo "Start to download VOCtrainval_11-May-2012.tar"
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
fi

if [ -f VOCtrainval_06-Nov-2007.tar ]
then
    echo "VOCtrainval_06-Nov-2007.tar found"
else
    echo "Start to download VOCtrainval_06-Nov-2007.tar"
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
fi

if [ -f VOCtest_06-Nov-2007.tar ]
then
    echo "VOCtest_06-Nov-2007.tar found"
else
    echo "Start to download VOCtest_06-Nov-2007.tar"
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
fi


# Extract the data.
tar -xvf VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtest_06-Nov-2007.tar
