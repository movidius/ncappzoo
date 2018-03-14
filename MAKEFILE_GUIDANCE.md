# Makefile Guidance 
See below for the guidence as to what must be in your project Makefiles when creating a pull request for a new project whether its an application or neural network.  The targets below are the minimum Makefile targets that are expected to be in your project, of course contributors are free to add any other targets that make sense for their particular project.

## Makefiles for applications
App Makefiles build source code and download or copy (from other repo subdirectories) any required files to the directory, as well as run your app. If your app needs content from other areas in the repo, like a network graph file, your Makefile should invoke the other Makefile to produce the content needed. You can use the apps/stream_infer/Makefile as the pattern to follow.  

### Makefile **required targets** for ncappzoo/apps directory: 
  - **make help** : Display make targets and descriptions.
  - **make data** : Download data (images, etc.) If no data download required create empty target.
  - **make deps** : Download/Prepare/compile networks.  If not needed create empty target.
  - **make all** : Prepares everything needed to run the application including other projects in the repository. Should not run application, should not popup GUI.
  - **make run** : Runs the application with default parameters if needed.
  - **make clean** : Removes all the files in this project directory that may get created when making or running this project.  Should not clean other projects in the repository.
  
 
 ## Makefiles for models
 Makefiles for Caffe or TensorFlow models should compile, profile, validate the neural network in the directory.  You can use the ncappzoo/caffe/GoogLeNet/Makefile as the pattern to follow.
 
 ### Makefile **required targets** for ncappzoo/caffe and ncappzoo/tensorflow directories:
  - **make help** : Display make targets and descriptions.
  - **make all** : creates a graph file for the network and does one of compile, check, or profile on it. Should not bring up GUI, or run lengthy program.
  - **make deps** : Download/Prepare networks.  If not needed create empty target
  - **make compile** : Run the NCSDK compiler to create a graph file.
  - **make profile** : Run the NCSDK profiler to display a profile of the network
  - **make check** : Run the NCSDK checker to validate the network
  - **make run** : Run a simple program demonstrating use of the compiled network 
  - **make clean** : Removes all the files in this project directory that may get created when making or running this project.  Should not clean other projects in the repository.
