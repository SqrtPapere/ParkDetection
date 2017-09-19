## Installation and Test

Requires following dependencies:
* python3
* OpenCV (3.1)
* Numpy
* [TensorFlow](https://github.com/samjabrahams/tensorflow-on-raspberry-pi)
* h5py (sudo apt-get install libhdf5-dev)
* Keras (pip3 install Keras)
* Pillow(pip3 install Pillow)
* picamera (pip3 install "picamera[array]")

Move directory “rasp-parkDetecting” to RaspberryPi and position it looking to the area to be monitored. 
Move directory “pc-parkDetectingg” to PC.

Connect to it by SSH -X. 

+ run: python3 waitCoordRASP.py on RaspberryPI
+ run: python3 selectCoordPC.py on PC
+ Select target of interest
+ press "c" key two times
+ run: python3 yoloRaspArray.py on raspberry
+ run: client.py on PC


## Contributors
Developed by [Francesco Pegoraro](https://github.com/SqrtPapere) and Daniele Giacomelli

## YOLO 

Joseph Redmon. Darknet: Open Source Neural Networks in C, 2013–2016. http://pjreddie.com/darknet/.
