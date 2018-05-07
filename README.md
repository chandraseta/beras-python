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

## Analysis
All testing is done individually for every background (black, red, and white). Each category is then splitted randomly with 80:20 training to testing data ratio.
### Canny
The following table contains accuracy of each background with varying k (number of neighbours) for the kNN algorithm using Canny edge detection.

| Background  | k=1 | k=3 | k=5 | k=10 | k=15 |
|:------------|:---:|:---:|:---:|:----:|:----:|
| Black       | 70,1% | 66,7% | 47,8% | 33,3% | 33,3% |
| Red         | 62,6% | 51,8% | 44,4% | 44,4% | 33,3% |
| White       | 33,3% | 33,3% | 33,3% | 33,3% | 33,3% |

Note:
- Accuracy for black background decreases as k increases. This might be caused by the small size of dataset. Larger k means the algorithm is more suspectible to overfit.
- Accuracy for red background is similar to accuracy for black background but slightly worse.
- Accuracy of white background with Canny is constant at 33,3% because, during preprocessing, the program generated completely black image. This means Canny failed to detect any edge of rice inside the image with white background. This also causes the rice program to always return **Class A** with any testing data.

### Black and White
The following table contains accuracy of each background with varying k (number of neighbours) for the kNN algorithm using grayscale image.

| Background  | k=1 | k=3 | k=5 | k=10 | k=15 |
|:------------|:---:|:---:|:---:|:----:|:----:|
| Black       | 81,1% | 70,1% | 62,6% | 41,1% | 33,3% |
| Red         | 70,1% | 66,7% | 55,6% | 50,2% | 55,6% |
| White       | 81,1% | 66,7% | 62,6% | 36,7% | 62,6% |

Note:
- Accuracy for black background decreases as k increases. This might be caused by the small size of dataset. Larger k means the algorithm is more suspectible to overfit.
- Accuracy for red background is decent, but not as good as the black background. The accuracy dips slightly at k=10 and increases again at k=15.
- Accuracy for white background, k=10 is somewhat an anomaly. The program mostly returns **Class C** during testing. This is quite confusing because when k=15, the accuracy goes back up to 62,6%.