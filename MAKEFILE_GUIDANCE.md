# Makefile Guidance 
See below for the guidence as to what must be in your project Makefiles when creating a pull request for a new project whether its an application or neural network.  The targets below are the minimum Makefile targets that are expected to be in your project, of course contributors are free to add any other targets that make sense for their particular project.

## Makefiles for applications
App Makefiles build source code, download or copy (from other repo subdirectories) any required files to the directory, as well as run your app. If your app needs content from other areas in the repo, like an optimized network, your Makefile should invoke the Makefile in the network's directory to produce the content needed. You can use the apps/simple_classifier_cpp/Makefile as the pattern to follow.  

### Makefile **required targets** for ncappzoo/apps directory: 
  - **make help** : Display make targets and descriptions.
  - **make data** : Download data (images, etc.) If no data is required this may be an empty target.
  - **make deps** : Download/Prepare/optimize networks.  If not needed create empty target.
  - **make all** : Prepares everything needed to run the application including other projects in the repository. Should not run application, should not popup GUI.
  - **make run** : Runs the application with some set of default parameters/settings/configuration.  'make run' must always execute the program without the need for other parameters.  Users expect to see some working output by just typing 'make run'
  - **make clean** : Removes all the files in this project directory that get created when making or running this project.  Should not clean other projects in the repository.  After running 'make clean' the only files in the directory should be those that are in the repository.
  - **make install-reqs**: Installs any required components for the application or gives instructions for installing those components.  If no other components are required then this target may be empty.  
 
 ## Makefiles for neural networks
 Makefiles for neural networks (in the **networks**, **caffe**, or **tensorflow**  directories) should optimize, and demonstrate the neural network in the directory.  You can use the ncappzoo/caffe/GoogLeNet/Makefile as the pattern to follow.
 
 ### Makefile **required targets** for ncappzoo/networks, ncappzoo/caffe, and ncappzoo/tensorflow directories:
  - **make help** : Display make targets and descriptions.
  - **make all** : makes the following targets: deps, compile_model
  - **make deps** : Download/Prepare networks to be optimized.  If not needed this may be an empty target
  - **make compile_model** : Run the OpenVINO toolkit's Model Optimizer tool to create an optimized OpenVINO IR neural network (.bin and .xml) for the network
  - **make run** : Run a simple program demonstrating the use of the optimized network.  This may invoke an application under the ncappzoo/apps directory hierarchy or from within this project's directory
  - **make clean** : Removes all the files in this project directory that may get created when making or running this project.  Should not clean other projects in the repository. After running 'make clean' the only files in the directory should be those that are in the repository.
- **make install-reqs**: Installs any required components for the neural network or gives instructions for installing those components.  If no other components are required then this target may be empty.

