# Beras - Python
Training and analysis of beras with Python3

## Requirements

### Python3
Before running the program, install Python 3.6:
* On Linux, open terminal and type: 
  ```
  sudo apt update
  sudo apt install python3.6
  ```
* [Anaconda](https://www.continuum.io/downloads) provides Python3.6 for Windows, macOS, and Linux too.

### OpenCV
Once you have Python3.6 installed, type the following commands:
```
pip3 install opencv-python
```
This will enable you to import `cv2`

### Matplotlib
Python3-tk is needed to use `pyplot` from `matplotlib`
```
sudo apt install python3-tk
pip3 install matplotlib
```
This will enable you to import `pyplot`

### Scikit-Learn
```
pip3 install scikit-learn
```

### Imutils
```
pip3 install imutils
```

## Dataset
Dataset subdirectories should be in the following format:
```
├── A
├── B
└── C
```
Where each subdirectory contains **jpg** files, naming does not affect the program.
Image in subdirectory A means the image belongs to class A in classification, and is also true for B and C.