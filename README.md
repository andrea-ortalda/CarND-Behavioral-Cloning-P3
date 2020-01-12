# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[image0]: ./write_up_images/center.png "Center Camera Image"
[image1]: ./write_up_images/left.png "Left Camera Image"
[image2]: ./write_up_images/right.png "Right Camera Image"
[image3]: ./write_up_images/nvidia.png "Nvidia Architecture"
[image4]: ./write_up_images/architecture.png "Architecture Used"
[image5]: ./write_up_images/training_workspace1.png "Training 1"
[image6]: ./write_up_images/training_workspace2.png "Training"
[video0]: ./write_up_images/video.gif "Final video"

<p align="center">
	<img src="/write_up_images/video.gif" alt="Final video"
	title="Final video"  />
</p>

### Overview

In this project, kowledge acquired during the course about deep neural networks and convolutional neural networks will be used to clone driving behavior. A model  trained, validated and tested using Keras. The model will output a steering angle to an autonomous vehicle.


The project contains five subitted files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of the vehicle driving autonomously around the track for at least one full lap)


### The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator.
* Summarize the results with a written report

#### 1. Use the simulator to collect data of good driving behavior 

Left Camera Image           | Center Camera Image             | Right Camera Image
:-------------------------:|:-------------------------:|:-------------------------:
![alt text][image1] |       ![alt text][image0] |      ![alt text][image2] 

