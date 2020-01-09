# Pedestrian detection using intel openvino pretrained model

This project detects pedestrain in given image of any dimension. Detection can be identified through bounding box sourrounding the pedestrain image. Bounding boxes can be seen in blue color. 
This is an end to end AI application running on intel edge hardware. I have deployed this app on Window 10 with intel i7 8th Gen CPU. There are pre and post image processing steps 
done before feeding image to model and plotting bounding box post model prediction. 

## Install

This project requires setup of intel openvino tooklit distribution setup on edge device. Please refer to below link reference for setup on Window 10 with intel i7 8th gen processor. 

https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_windows.html

For setup on other intel edge devices refer to below link. 

https://docs.openvinotoolkit.org/latest/index.html

Intel openvino toolkit also supports below libraries required to run this solution 

- [NumPy]
- [CV2]
- [openvino.inference_engine]

##  Prerequisites 
Below are prerequisites for this project 
- Python 3.6.xxx version to support libraries [os, sys, logging]
- Intel Openvino toolkit setup
- Cmake installation 
- Visual Studio 2019

For python and cmake installation, follow the instructions on openvino documentation

## Execution 

This app is only supported through command line execution. Command line accepts following arguments 
- Path to input image 
- Path to Openvino pretrained model XML file 
- Path to CPU Extension Lib

Here are some sample commands to run this app. 

python app.py -i input_images/man-crossing-road.jpg -m intel/pedestrian-detection-adas-0002/FP16/pedestrian-detection-adas-0002.xml -c "C:\Program Files (x86)\IntelSWTools\openvino\inference_engine\bin\intel64\Release\cpu_extension_avx2.dll"

python app.py -i input_images/back-view-photo-of-man-standing-on-sidewalk-looking-to-his.jpg -m intel/pedestrian-detection-adas-0002/FP16/pedestrian-detection-adas-0002.xml -c "C:\Program Files (x86)\IntelSWTools\openvino\inference_engine\bin\intel64\Release\cpu_extension_avx2.dll"
 

## Data
Input images are provided in input_images folder. Pedestrains are detected on these images with rectangular bounding box and generate output image can be located in output_images folder. 


## Data Preprocessing 

Pre-processing: Image input of any dimension is resized to dimenions of width and height defined by model input. Transponse is done to reorder the color channel and atlast image is reshaped to match the 
dimension of input accepted by model. 

Post-processing: Bounding boxes with confidence greater than 0.8 are filtered and further processed to draw on input image. Before plotting coordinates of bounding box are scaled as per
input image dimensions.  


## Detection Model 
Pretrained Intel Openvino Model used in this project - pedestrian-detection-adas-0002
Model can be downloaded from Openvino website - https://docs.openvinotoolkit.org/latest/_models_intel_pedestrian_detection_adas_0002_description_pedestrian_detection_adas_0002.html


## Observation
Bounding box can be clearly seen on pedestrain on givem image. There are some limitations to this detection listed below 
- pedestrain facing backwards are not predicted successfully. 
- pedestrain far behind in image are also ignored as confidence threshold is set too high = 0.8. Lower confidence can increase probability of detecting pedestrains far behind in this image.


## Miscellanous 
For more details on how model algorithm works, please refer to below link. 
- https://arxiv.org/pdf/1704.04861.pdf (MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications)
- https://arxiv.org/abs/1512.02325 (SSD: Single Shot MultiBox Detector)