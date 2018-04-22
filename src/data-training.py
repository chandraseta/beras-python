import cv2
import glob
import numpy as np
import random

from sklearn.neighbors import KNeighborsClassifier

def get_files(img_type, grade, training_set_size):
    files = glob.glob('../data/{}/{}/*'.format(img_type, grade))
    random.shuffle(files)
    training = files[:int(len(files) * training_set_size)]
    prediction = files[-int(len(files) * (1 - training_set_size)):]

    return training, prediction

def img_to_feature_vector(image):
    return image.flatten()

def extract_color_histogram(image, bins=(8,8,8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None, bins, [0, 180, 0, 256, 0, 256])

    cv2.normalize(hist, hist)

    return hist.flatten()

def generate_sets(img_type, grades):
    training_raw_data = []
    training_features = []
    training_labels = []
    prediction_raw_data = []
    prediction_features = []
    prediction_labels = []

    for grade in grades:
        training, prediction = get_files(img_type, grade, 0.8)

        for item in training:
            image = cv2.imread(item)
            pixels = img_to_feature_vector(image)
            hist = extract_color_histogram(image)

            training_raw_data.append(pixels)
            training_features.append(hist)
            training_labels.append(grades.index(grade))

        for item in prediction:
            image = cv2.imread(item)
            pixels = img_to_feature_vector(image)
            hist = extract_color_histogram(image)

            prediction_raw_data.append(pixels)
            prediction_features.append(hist)
            prediction_labels.append(grades.index(grade))

    return training_raw_data, training_features, training_labels, prediction_raw_data, prediction_features, prediction_labels

def train_data(img_type, grades, k=3):
    training_raw_data, training_features, training_labels, prediction_raw_data, prediction_features, prediction_labels = generate_sets(img_type, grades)

    print('Using KNN algorithm with raw pixel')
    # knn = cv2.ml.KNearest_create()
    knnRaw = KNeighborsClassifier(n_neighbors=k)
    
    print('Size of training set is {} images'.format(len(training_labels)))
    # knn.train(training_data[0], cv2.ml.ROW_SAMPLE, training_labels[0])
    knnRaw.fit(training_raw_data, training_labels)

    print('Finished training')
    accRaw = knnRaw.score(prediction_raw_data, prediction_labels)

    print("Accuracy using raw pixel: {}%".format(accRaw*100))

    print('Using KNN algorithm with histogram')
    knnHist = KNeighborsClassifier(n_neighbors=k)

    print('Size of training set is {} images'.format(len(training_labels)))
    # knn.train(training_data[0], cv2.ml.ROW_SAMPLE, training_labels[0])
    knnHist.fit(training_features, training_labels)

    print('Finished training')
    accHist = knnHist.score(prediction_raw_data, prediction_labels)

    print("Accuracy using histogram: {}%".format(accHist*100))


if __name__ == '__main__':
    grades = ['A', 'B', 'C']

    yes = {'yes', 'ye', 'y', ''}
    no = {'no', 'n'}
    choice = input('Use canny?[Y/n] ').lower()

    while choice not in yes and choice not in no:
        print('Sorry, did not quite catch that')
        choice = input('Use canny?[Y/n] ').lower()

    canny_mode = False
    if choice in yes:
        canny_mode = True
        print("Canny mode activated")
    else:
        print("Black and White mode activated")

    img_type = 'canny' if canny_mode else 'bw'

    k = input('Number of neigbours: ')

    while not k.isdigit() or int(k)==0:
        print('Invalid number, must be a positive')
        k = input('Number of neigbours: ')

    correct, percentage = train_data(img_type, grades, int(k))
    print('Processed ', correct, ' data correctly')
    print('Accuracy ', percentage, '%')
