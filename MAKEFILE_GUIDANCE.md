# Makefile Guidance 
See below for the guidence as to what must/should be in your Makefiles when creating a pull request.

## Makefiles in the apps directory
App Makefiles build source code and download or copy (from other repo subdirectories) any required files to the directory, as well as run your app. If your app needs content from other areas in the repo, like a network graph file, your Makefile should invoke the other Makefile to produce the content needed. You can use the apps/stream_infer/Makefile as the pattern to follow.  

### App Makefile **required targets**: 
  - **make help** : Display make targets and descriptions.
  - **make data** : Download data (images, etc.) If no data download required create empty target.
  - **make deps** : Download/Prepare/compile networks.  If not needed create empty target.
  - **make all** : Prepares everything needed to run the application including other projects in the repository. Should not run application, should not popup GUI.
  - **make run** : Runs the application with default parameters if needed.
  - **make clean** : Removes all the files in this project directory that may get created when making or running this project.  Should not clean other projects in the repository.
  
 
 ## Makefiles in the caffe or tensorflow directories
 Makefiles for Caffe or TensorFlow models should compile, profile, validate the neural network in the directory.
 
 ### Model Makefile **required targets** are:
  - **make help** : Display make targets and descriptions.
  - **make all** : runs compile, profile, and check for the network.
  - **make deps** : Download/Prepare networks.  If not needed create empty target
  - **make compile** : Run the NCSDK compiler to create a graph file.
  - **make profile** : Run the NCSDK profiler to display a profile of the network
  - **make check** : Run the NCSDK checker to validate the network
  - **make run** : Run a simple program demonstrating use of the compiled network 
  - **make clean** : Removes all the files in this project directory that may get created when making or running this project.  Should not clean other projects in the repository.
