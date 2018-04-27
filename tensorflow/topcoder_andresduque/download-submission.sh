#!/bin/bash
fileid="1NDLvd9jfoBx-MYTcT0M5m1XFMdAHjPc6"
filename="submission_andresuduque.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
