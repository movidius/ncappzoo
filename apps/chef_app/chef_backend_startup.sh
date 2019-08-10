#!/bin/bash

echo Installing prerequisites...
sudo pip3 install redis
sudo pip3 install celery

echo Initiating brokers and workers...
gnome-terminal -e "./run-redis.sh"
gnome-terminal -e "celery worker -A openvino_backend.celery --loglevel=info"

echo Initiating backend server...
python3 openvino_backend.py
