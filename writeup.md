## Writeup 
---

## Vehicle Detection Project

**I have implemented Tiny YOLOv2, a Deep learning based approach that can detect objects in images**  

The (revised) goals / steps of this (Deep Learning) project are the following:

* Implement a Keras model of tiny YOLOv2
* Use pre-trained weights for the model and load them
* Test that the network works for the given test images
* Fine-tune parameters and perform pre-processing of image to prevent false positives
* Run the pipeline on a video stream and estimate a bounding box for vehicles detected.

[//]: # (Image References)
[yolo]: ./output_images/yolo2.png
[testimg]: ./output_images/test1_detected.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
---
**I have attempted to follow the given rubric as closely as possible for my DL-based approach**

## YOLOv2

The papers for YOLO can be found at [YOLOv2](https://arxiv.org/abs/1612.08242) and [YOLOv1](https://arxiv.org/abs/1506.02640). 

### Summary of how YOLO works
Other systems use classifiers or localizers to detect objects. This classifier is applied to multiple parts of the image at different scales. When a region has a high score, it is considered a detection. 

YOLO takes a different approach - it applies a single network to the whole image. The network divides the image into regions and predicts bounding boxes and probabilities of each region. The bounding boxes are weighted by the predicted probabilities. 

![yolo]

YOLO has many advantages over other systems, but importantly, for us, it is very fast and requires only one evaluation on a test image. On a good GPU tiny YOLO performs fast enough for real time applications. 



### Changes in YOLOv2
YOLOv2 uses a few tricks to improve training and increase performance over YOLOv1. It uses a fully-convolutional model and adjusts priors on bounding boxes instead of predicting the width and height outright.

### Tiny YOLO

Tiny YOLO is based off of the [Darknet reference network](https://pjreddie.com/darknet/imagenet/#reference) and is much faster but less accurate than the normal YOLO model. On a fast GPU it runs at > 200 FPS. 

The model of the network is the same provided by the `.cfg` file at [YOLO](https://pjreddie.com/darknet/yolo/). The summary of the model is given in the table below

Layer (type)                 | Output Shape              | Param #   
--- | --- | ---
conv2d_1 (Conv2D)            | (None, 416, 416, 16)      | 432       
batch_normalization_1 (Batch | (None, 416, 416, 16)      | 64        
leaky_re_lu_1 (LeakyReLU)    | (None, 416, 416, 16)      | 0         
max_pooling2d_1 (MaxPooling2 | (None, 208, 208, 16)      | 0         
conv2d_2 (Conv2D)            | (None, 208, 208, 32)      | 4608      
batch_normalization_2 (Batch | (None, 208, 208, 32)      | 128       
leaky_re_lu_2 (LeakyReLU)    | (None, 208, 208, 32)      | 0         
max_pooling2d_2 (MaxPooling2 | (None, 104, 104, 32)      | 0         
conv2d_3 (Conv2D)            | (None, 104, 104, 64)      | 18432     
batch_normalization_3 (Batch | (None, 104, 104, 64)      | 256       
leaky_re_lu_3 (LeakyReLU)    | (None, 104, 104, 64)      | 0         
max_pooling2d_3 (MaxPooling2 | (None, 52, 52, 64)        | 0         
conv2d_4 (Conv2D)            | (None, 52, 52, 128)       | 73728     
batch_normalization_4 (Batch | (None, 52, 52, 128)       | 512       
leaky_re_lu_4 (LeakyReLU)    | (None, 52, 52, 128)       | 0         
max_pooling2d_4 (MaxPooling2 | (None, 26, 26, 128)       | 0         
conv2d_5 (Conv2D)            | (None, 26, 26, 256)       | 294912    
batch_normalization_5 (Batch | (None, 26, 26, 256)       | 1024      
leaky_re_lu_5 (LeakyReLU)    | (None, 26, 26, 256)       | 0         
max_pooling2d_5 (MaxPooling2 | (None, 13, 13, 256)       | 0         
conv2d_6 (Conv2D)            | (None, 13, 13, 512)       | 1179648   
batch_normalization_6 (Batch | (None, 13, 13, 512)       | 2048      
leaky_re_lu_6 (LeakyReLU)    | (None, 13, 13, 512)       | 0         
max_pooling2d_6 (MaxPooling2 | (None, 13, 13, 512)       | 0         
conv2d_7 (Conv2D)            | (None, 13, 13, 1024)      | 4718592   
batch_normalization_7 (Batch | (None, 13, 13, 1024)      | 4096      
leaky_re_lu_7 (LeakyReLU)    | (None, 13, 13, 1024)      | 0         
conv2d_8 (Conv2D)            | (None, 13, 13, 1024)      | 9437184   
batch_normalization_8 (Batch | (None, 13, 13, 1024)      | 4096      
leaky_re_lu_8 (LeakyReLU)    | (None, 13, 13, 1024)      | 0         
conv2d_9 (Conv2D)            | (None, 13, 13, 125)       | 128125    
activation_1 (Activation)    | (None, 13, 13, 125)       | 0         
reshape_1 (Reshape)          | (None, 13, 13, 5, 25)     | 0         
---
Total params: 15,867,885  
Trainable params: 15,861,773  
Non-trainable params: 6,112

### Pre-training
YOLO is trained on the VOC dataset and COCO dataset. Weights are available at [this link](https://pjreddie.com/darknet/yolo/). I used the weights for Tiny YOLOv2 trained on the VOC dataset. 

The VOC dataset has 20 classes, of which, cars is one (it is the 6th class). This allows us to directly use the network to perform detection of cars. 

### Converting the weights
The weights provided are in a `.weights` format. To convert them to a `.h5` format I used the converter provided by `YAD2k` with instructions [here](https://github.com/allanzelener/YAD2K) 

### Image Pre-processing
The image is cropped to focus on the region of interest. It is done to basically ignore the top half of the image and some portion on the left. 

Additionally, each image pixel is normalized to have values between -1 and 1.


### Class confidence threshold

Output from grid cells below a certain threshold (0.3) of class probability are rejected. This eliminates false positives

### Reject overlapping (duplicate) bounding boxes

If multiple bounding boxes overlap and have an intersecting area greater than 25% of the combined area, then we keep the box with the highest class score and reject the other box(es).

### Test Output
The output for an image is given below. The confidence of the network that it is a car is given in green near the bounding box. 

![testimg]
---

### Video Implementation
The video can be viewed by clicking on the thumbnail below

[![Advanced Lane Finding](https://img.youtube.com/vi/mEdwV9ww3Rw/0.jpg)](https://www.youtube.com/watch?v=mEdwV9ww3Rw)



---

### Discussion

The DL-based method has the advantage of being more robust and require less feature engineering and hand-holding. However, if we were to build this network from scratch it would require a lot of training data and good compute power to reach this level of performance. 

Performance can be further improved by using the methods of transfer learning and making the network only detect cars. 

It runs quite quickly on my Macbook Pro 13" with an i5 processor and 8gb of RAM. About 0.5 frames per second was the speed of processing.  My laptop does not have GPU support. The authors of the paper say that with a good GPU we can achieve results that are good enough for real-time applications (> 30fps).
