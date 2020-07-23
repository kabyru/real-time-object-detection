# Real Time Object Detection
#### A driver program written to evaluate custom OD models in real time.
##### Written by Kaleb Byrum for Raytheon Technologies.

## Introduction
This program aims to simplify the use of custom TensorFlow object detection models in a real-time setting. OpenCV is used to input a webcam frame, which is then used as input as evaluation of an object detection model. The results of the evaluation are returned in the same window as the webcam input, creating a real-time implementation of object detection.

## Setup
You will need the following Python packages to use this program:
### Required Python packages
#### Install using 'pip install ...'

 - Numpy
 - TensorFlow (Use version 2.0, 'pip install tensorflow')
 - opencv-python
 - imutils
 - matplotlib
 - pillow
 - IPython
 - pyyaml
 - The Object Detection package, created using the TensorFlow models directory.

### Step-by-Step Installation

 1. Install Python. You may choose to either install the vanilla Python 3.7 [here](https://www.python.org/downloads/release/python-378/) or install the Anaconda distribution for Python 3.7 [here.](https://www.anaconda.com/products/individual) *The latter is recommended.*
 2. Install the above required Python packages. *Many of these packages come included when you install Anacanda, but do not if you install vanilla Python.*
 You will need to build the object-detection package [following these instructions.](https://colab.research.google.com/drive/1UEiJAnyp4gxukRZgEmHA9Qvh21vXyJ6k?usp=sharing)
 3. Download the custom object detection model you want to use. Either choose from the [TensorFlow Object Detection Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) or make sure you have one locally on your system.
#### If you received this program from me, custom models will be sent separately since they are not publicly released.
4. Open the **file-paths.yaml** file, and make sure the file paths lead correctly to the object detection model, and label map. Configure the camera resolution in this file as well.
	##### Note: This release omits using image segmentation, simply because our purposes do not require it. However, it can be easily integrated by uncommenting it out.
5. At this point, the program should be ready to launch! You can launch the file using the command line
> python ./main.py

A demo video will be provided that demonstrates expected results from this program using a custom model.