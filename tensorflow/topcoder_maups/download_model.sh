#!/bin/bash

curl -s -c /tmp/cookies "https://drive.google.com/uc?export=download&id=10_KVlnbLg38KRpue08I3CkD7GI5JDTfI" > /tmp/tmp.html
curl -s -L -b /tmp/cookies "https://drive.google.com$(cat /tmp/tmp.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > model.tgz
rm /tmp/cookies /tmp/tmp.html

