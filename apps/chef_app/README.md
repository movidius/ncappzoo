# Devmesh Backend

## Demo

![Chef.ai in action!](./chef_ai.gif)

## How to use the app:
1. Take a picture of the ingredients that you have or upload a picture
2. Detect the food
3. See what is detected, add changes as needed
4. Get recipes
5. See recipes and choose from 10 max
6. Repeat!

## Important Details:
- Backend Stack: Flask, Celery, Redis
- The Flask backend is implemented asynchronously to accommodate for heavy traffic.
- Use of task IDs and interval function calls are used to track status/result
  1. Use initial AJAX call to get the task ID
  2. At interval, use another AJAX call using the task ID to grab status
  3. End interval checking once AJAX call responds with success/failure
- The test react app demonstrates how to properly use the Flask backend
- Object detection model based on images from the internet, labeled by Christian
  - Mimics the general workflow of a ML software developer developing on Tensorflow

## Notable Dependencies:
- Backend:
  - flask
  - celery
  - openCV
  - OpenVINO 2019 R1.1
  - numpy
  - base64
  - PIL.Image
  - io
  - redis
- Frontend:
  - React dependencies
  - axios
  - react-webcam

## Setup

### Installing Dependencies
1. pip3 or pip install all the python dependencies for the backend
2. Set up redis by following the installation in the backend setup section.
2. `npm install` in the `test_react_app` folder to install all the dependencies for the react app.
3. Download weights for food model and gesture model
  - Food Model FP32: https://drive.google.com/open?id=1nDDaEpm7hNG8eR4vXBATMpWRrOL-4-jq
  - Food Model FP16: https://drive.google.com/open?id=1eC-ZKpiSOW8pHZ1m3rIbnMfcKN2Hf1gI
  - In the case that you don't like my models, please feel free to train your own models and plug them into the app. My recommendation is to follow the tutorial links below to train using Tensorflow's Object Detection API.

### Procedure for starting application:
1. open four terminals
2. first terminal -> `$ redis-server`
3. second terminal -> `$ cd <INSTALL_DIR>/devmesh_backend/ && celery worker -A openvino_backend.celery --loglevel=info`
4. third terminal -> `python3 openvino_backend.celery`
5. Make a config.js file and add the necessary requirements such as app_id, app_key, and Raspberry Pi backend url.
5. fourth terminal -> `cd <INSTALL_DIR>/devmesh_backend/test_react_app && npm start`

### Additional Details for config.js
1. Make an account on dataplicity and go through their tutorial to setup your raspberry pi for access via the internet
2. Use the special url and copy and paste that into the first key `BACKEND_URL`
3. Go to Edamam API's website and register for an account. You will need to sign up for their recipe API as well as make an application that uses the recipe API.
4. Copy the app ID and app key into the remaining key-value pairs, respectively

### Common Errors:
- Dealing with ELIFECYCLE Errors?
  1. `sudo npm cache clean --force`
  2. `sudo rm -rf node_modules`
  3. `sudo npm install --save`
  3. `npm start`
- Dealing with modules not being found?
  1. Pay attention to what you used to install dependencies. Pip and pip3 might not install to the python you might be expecting. In addition, using sudo rather --user may also lead to some referencing problems. Make sure to stay consistent!

## References

### Frontend References:
- https://www.npmjs.com/package/react-webcam
- https://www.npmjs.com/package/react-images-upload

### Backend Setup:
- https://blog.miguelgrinberg.com/post/using-celery-with-flask
- https://redis.io/topics/quickstart

### Object detection model tutorial:
- https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html
- https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
- https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/issues/184

## Past Research

### OCR:
- ???

### PixelNet Detection:
- https://github.com/banderlog/open_model_zoo/blob/24ca50034555721d876c8314ad36b4d6b03bf321/demos/python_demos/text_detection_demo.py

### EAST Text Detection:
- https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
- https://medium.com/@tomhoag/opencv-text-detection-548950e3494c
- https://bitbucket.org/tomhoag/opencv-text-detection/src/master/
