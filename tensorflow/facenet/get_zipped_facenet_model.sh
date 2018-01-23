#! /bin/bash

echo "Attempting to download 20170512-110547.zip from this url:"
echo "https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk"

rm -f ./cookies.txt
touch ./cookies.txt
wget --load-cookies ./cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ./cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B5MzpY9kBtDVZ2RpVDYwWmxoSUk' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0B5MzpY9kBtDVZ2RpVDYwWmxoSUk" -O 20170512-110547.zip && rm -rf ./cookies.txt


