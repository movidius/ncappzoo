#!/bin/bash

python3 -c "from tkinter import*" > /dev/null 2>&1
if [ $? -eq 0 ] ;
then
	echo "";
	echo "tkinter already setup.";
	echo "";
	exit 0;
else
	echo "";
	echo "tkinter is not installed.";
	echo "Proceeding to install python3-tk";
	sudo -H apt-get install python3-tk;
	echo "";
	exit 0;
fi
