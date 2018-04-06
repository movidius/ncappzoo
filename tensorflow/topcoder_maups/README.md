# Intel Movidius Marathon Match - Solution Description

For details on this competition, please visit the [Intel Movidius Marathon Match website on TopCoder](https://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=17058&pm=14775).

## 1. Introduction

Tell us a bit about yourself, and why you have decided to participate in the contest.

- Name: Mauricio Pamplona Segundo
- Handle: maups
- Placement you achieved in the MM: 2nd place
- About you: I am a professor at the Federal University of Bahia, Brazil, and I'm temporarily working as a postdoctoral scholar at the University of South Florida, Tampa. I have worked with research in computer vision for the past 12 years, but just started working with deep learning recently.
- Why you participated in the MM: One of the main reasons for me to work with computer vision is being able to create viable real-world applications. The Neural Compute Stick seems to be a promising way of achieving that using deep learning.

## 2. Solution Development 

How did you solve the problem? What approaches did you try and what choices did you make, and why? Also, what alternative approaches did you consider?

- My solution to this challenge was to fine tune a small portion of layers from different CNNs pretrained on ILSVRC-2012-CLS to check which one would give the best results.
- The CNN architectures I tried were: Inception V1, Inception V4, Inception-ResNet-v2 and VGG 16.
- I also tried to train a SqueezeNet from scratch to check if the gain in speed would compensate the loss in accuracy, but it did not.

## 3. Final Approach

Please provide a bulleted description of your final approach. What ideas/decisions/features have been found to be the most important for your solution performance:

- Download all images of the 200 synsets from [ImageNet](http://www.image-net.org/) and crop the biggest centered square in each of them. Split the data in training and validation sets (2000 images for validation in order to obtain good estimates of the performance in the test set).
- Use Inception V4 code and weights from [here](https://github.com/tensorflow/models/tree/master/research/slim) as start point for training.
- Retrain the last layer of the network ('InceptionV4/Logits') for 100 epochs using random data augmentation (horizontal flips, contrast changes, and shifts in x and y directions), batch size equal to 256 and learning rate equal to 0.1\*0.98^epoch. Zoom is not used at this point to speed up the process.
- Use the obtained model to retrain the last layer ('InceptionV4/Logits') and the last inception block ('InceptionV4/Mixed_7d') for 15 epochs using random data augmentation (horizontal flips, constrast changes, shifts in x and y directions and zoom), batch size equal to 256 and learning rate equal to 0.1\*0.98^epoch.
- Use the obtained model to retrain the last layer ('InceptionV4/Logits') and the last two inception blocks ('InceptionV4/Mixed_7d' and 'InceptionV4/Mixed_7c') for 2 epochs using random data augmentation (horizontal flips, constrast changes, shifts in x and y directions and zoom), batch size equal to 256 and learning rate equal to 0.1\*0.98^epoch.
- The above training process was done in a cascade of steps to avoid overfitting.

## 4. Open Source Resources, Frameworks and Libraries

Please specify the name of the open source resource along with a URL to where it’s housed and it’s license type:

- Python 2.7, https://www.python.org/, (PSFL)
- C++11, https://gcc.gnu.org/ (GPL)
- OpenCV, https://opencv.org/ (BSD 3-clause)
- Intel Movidius Neural Compute SDK, https://developer.movidius.com/ (LICENSE)
- TensorFlow, https://www.tensorflow.org/ (Apache License 2.0)

## 5. Potential Algorithm Improvements

Please specify any potential improvements that can be made to the algorithm:

- The cascade of training steps can be extended to cover more layers.
- Investigate other CNN architectures.

## 6. Algorithm Limitations

Please specify any potential limitations with the algorithm:

- As any other CNN, it is susceptible to adversarial attacks.
- Some parameters are not trained with single-precision floats, so the results may not be exactly the same when running in the Neural Compute Stick.

## 7. Deployment Guide

Please provide the exact steps required to build and deploy the code:

- Install OpenCV and the Intel Movidius Neural Compute SDK. Then follow instructions from:

```
$ make help
```

## 8. Final Verification

Please provide instructions that explain how to train the algorithm and have it execute against sample data:

- Follow instructions from:

```
$ make help
```

## 9. Feedback

Please provide feedback on the following - what worked, and what could have been done better or differently?

- Problem Statement - The problem statement was lacking details regarding what was expected in the README file and in the final submission.
- Data - Using images from an open database such as ImageNet in the hidden test set was not a good strategy.
- Contest - Everything went well, but some important information could only be found in the forum. Maybe there should be a way of highlighting specific forum posts. 
- Scoring - It would be nice to have some kind of feedback when the score is 0, as in this case there were 3 different possibilities: invalid submission, maximum logloss value exceeded, and time limit exceeded.


