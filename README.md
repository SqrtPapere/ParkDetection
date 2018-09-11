## Installation and Test

Requires following dependencies:
* python3
* OpenCV (3.1)
* Numpy
* [TensorFlow](https://github.com/samjabrahams/tensorflow-on-raspberry-pi)
* h5py (`sudo apt-get install libhdf5-dev`)
* Keras (`pip3 install Keras`)
* Pillow(`pip3 install Pillow`)
* picamera (`pip3 install "picamera[array]"`)

Copy directory `rasp-parkDetecting` into RaspberryPi. Position piCamera looking to the area to be monitored. 
Move directory `pc-parkDetecting` to PC.

Connect to it by SSH. 

+ run: `python3 waitCoordRASP.py` on RaspberryPI
+ run: `python3 selectCoordPC.py` on PC
+ Select target of interest
+ press "c" key two times
+ run: `python3 yoloRaspArray.py` on raspberry
+ run: `python3 client.py` on PC

## Results
Every time a change of state happens, a pic will be taken:

[![4.png](https://github.com/SqrtPapere/ParkDetection/blob/master/img/1.jpg)](https://github.com/SqrtPapere/ParkDetection/blob/master/img/1.jpg)[![2.png](https://github.com/SqrtPapere/ParkDetection/blob/master/img/1.jpg)](https://github.com/SqrtPapere/ParkDetection/blob/master/img/1.jpg)[![3.png](https://github.com/SqrtPapere/ParkDetection/blob/master/img/1.jpg)](https://github.com/SqrtPapere/ParkDetection/blob/master/img/1.jpg)[![1.png](https://github.com/SqrtPapere/ParkDetection/blob/master/img/1.jpg)](https://github.com/SqrtPapere/ParkDetection/blob/master/img/1.jpg)

## Contributors
Developed by [Francesco Pegoraro](https://github.com/SqrtPapere) and [Daniele Giacomelli](https://github.com/DanieleGiacomelli)

## YOLO 

Joseph Redmon. Darknet: Open Source Neural Networks in C, 2013–2016. http://pjreddie.com/darknet/.
