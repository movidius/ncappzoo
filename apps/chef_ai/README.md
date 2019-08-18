# Chef.ai

## Demos

### Chef.ai Webapp with Flask Templates/Vanilla JS
![Chef.ai in action!](./chef_ai.gif)

The original [Chef.ai](https://github.com/fcr3/chef_devmesh) webapp with React.js
as the frontend can in the link provided.

## Introduction
This app demonstrates how to use the food ssd inception v2 model featured in the ncappzoo's networks directory. This app features a frontend and backend, and the OpenVINO inference code is plugged into the backend. Once users upload the data such that the frontend holds the data in memory, users can then run inference on the data, spit out a prediction on what food is available in the image data, and use an external API to generate recipes using those foods.

## How to use the app:
1. Take a picture of the ingredients that you have or upload a picture
2. Detect the food
3. See what is detected, add changes as needed
4. Get recipes
5. See recipes and choose from 10 max
6. Repeat!

## Setup
All setup/startup is streamlined through the make commands!

### Start the application
```
$ # Set the OpenVINO environment by sourcing <openvino install dir>/bin/setupvars.sh
$ make install-reqs
$ make run
```

### Makefile
- **make all** - does **make data** and **make deps**
- **make data** - downloads test data
- **make deps** - downloads the optimized OpenVINO models
  - **make deps_FP16** and **make deps_FP32** are subroutines of **make deps**
  - Commands download their respective models based on precision
- **make run** - runs the flask backend and opens the webpage. Plug in your Intel NCS 2 in order to use `Detect (MYRIAD)`!
- **make install-reqs** - downloads required packages and installs them
- **make uninstall-reqs** - uninstalls required packages
- **make clean** - deletes data/models
- **make help** - lists what the commands above do

### How to set up Edamam Account to use API
1. Go to Edamam API's website and register for an account. You will need to sign up for their recipe API as well as make an application that uses the recipe API.
2. Copy the app ID and app key into the remaining key-value pairs, respectively into the beginning portion of the script section in `templates/index.html`. If you have trouble, do a text search for `CONFIGS` to find the config keys.

## Details:
- Backend Stack: Flask
- Webpage is served via root route (127.0.0.1:5000)
  - Requests to the backend are done via fetch HTTP requests
- Synchronous Backend
- You can find original repository [here](https://github.com/fcr3/chef_devmesh).

## Notable Dependencies:
- Backend:
  - flask
  - PIL.Image
  - numpy
  - base64
  - openCV
  - OpenVINO 2019 R2
- Fronend:
  - Vanilla Js (whatever Javascript comes with the browser)
  - Regular HTML/CSS
