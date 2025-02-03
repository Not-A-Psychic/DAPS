# DAPS
Using MRCNN to detect available parking spaces

## AIM:
To detect empty and occupied parking spaces by using MRCNN on tensorflow 1.3.0

## Introduction:
As cities grow, effectively managing parking has become an important part of smart city infrastructure. Manual checks or simple sensor Systems are two examples of old-fashioned ways to keep an eye on parking lots. They are often expensive, require a lot of work, and make mistakes. Better ways to keep an eye on parking spots are now possible thanks to progress in computer vision and deep learning, especially in object recognition and segmentation algorithms.

## Solution:
Inspired from https://github.com/matterport/Mask_RCNN.
To deal with this problem, we thought of a solution, one which uses cameras already existing on parking lots. By segmenting occupied and available parking spaces, drivers can see before hand if there are any parking spots available to park.

## Libs/Tech used:
This project was performed purely from a learning point of view and done with the following technologies/libraries:
- Python 3.5 (anaconda)
- TensorFlow 1.3.0

There are a host of other libraries used, which are listed in the inf.yaml and train.yaml anaconda backups.

## Additional notes:
To change the backbone from resnet101, learning rate, momentum weight decay or any other hyperpapameters, checkout train/mrcnn/config.py. these are some code lines that might be useful for your experimentation:
backbone - 55
learning rate - 181
momentum - 182
weight decay - 185

To get started with this project, follow along this tutorial at digital ocean https://www.digitalocean.com/community/tutorials/mask-r-cnn-in-tensorflow-2-0.