from flask import Flask, jsonify, request, redirect
from werkzeug.utils import secure_filename

import cv2
import glob
import numpy as np
import os
import random

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# Initiate Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check for allowed extensions (png, jpg, jpeg)
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Normalize image by resizing (500 x 500) and preprocessing
# Method == 1: Canny
# Method == 2: Grayscale
# Method == 3: Resize only
def normalize_image(image, method):
    image = cv2.resize(image, (500, 500))

    if (method == 1):
        # Denoising
        image = cv2.fastNlMeansDenoisingColored(image, None, 25, 7, 9)

        # Add blur
        image = cv2.GaussianBlur(image, (3,3), 0)

        # Automatically calculate lower and upper bound
        sigma = 0.33
        grey = np.median(image)
        lower = int(max(0, (1.0 - sigma) * grey))
        upper = int(min(255, (1.0 + sigma) * grey))

        # Transform image using Canny
        image = cv2.Canny(image, lower, upper)
    elif (method == 2):
        # Transform image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image

# Get file
def get_files(img_type, grade, training_set_size):
    files = glob.glob('./data/{}/{}/*'.format(img_type, grade))
    random.shuffle(files)
    training = files[:int(len(files) * training_set_size)]
    prediction = files[-int(len(files) * (1 - training_set_size)):]

    return training, prediction

# Return a flat list of image's pixels
def img_to_feature_vector(image):
    return image.flatten()

# Extract histogram from image
def extract_color_histogram(image, bins=(8,8,8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None, bins, [0, 180, 0, 256, 0, 256])

    cv2.normalize(hist, hist)

    return hist.flatten()

# Generate training and testing sets
def generate_sets(img_type, grades):
    training_raw_data = []
    training_labels = []
    prediction_raw_data = []
    prediction_labels = []

    for grade in grades:
        training, prediction = get_files(img_type, grade, 1)

        for item in training:
            image = cv2.imread(item)
            pixels = img_to_feature_vector(image)

            training_raw_data.append(pixels)
            training_labels.append(grades.index(grade))

        for item in prediction:
            image = cv2.imread(item)
            pixels = img_to_feature_vector(image)

            prediction_raw_data.append(pixels)
            prediction_labels.append(grades.index(grade))

    return training_raw_data, training_labels, prediction_raw_data, prediction_labels

# PCA, to be used with eigen vector, not kNN
def pca(images, num_component):
    mean, eigen_vector = cv2.PCACompute(images, mean=None, maxComponents=num_component)
    return mean, eigen_vector

# Predict image based on img_type and k
# img_type: canny or bw (black and white)
# k: amount of neighbours
def predict_image(image, img_type, grades, k=1):
    training_raw_data, training_labels, prediction_raw_data, prediction_labels = generate_sets(img_type, grades)
    training_raw_data = np.array(training_raw_data, dtype='f')
    training_labels = np.array(training_labels, dtype='f')

    prediction_raw_data = []

    pixels = img_to_feature_vector(image)

    # Compensate from 1 (Grayscale) to 3 channel (RGB)
    opt_pixel = []
    for pixel in pixels:
        opt_pixel.append(pixel)
        opt_pixel.append(pixel)
        opt_pixel.append(pixel)

    prediction_raw_data.append(opt_pixel)
    prediction_raw_data = np.array(prediction_raw_data, dtype='f')

    # kNN with pixel values
    knn = cv2.ml.KNearest_create()
    knn.train(training_raw_data, cv2.ml.ROW_SAMPLE, training_labels)
    ret, results, neighbours, dist = knn.findNearest(prediction_raw_data, k)

    return results[0]

@app.route("/predict/",methods=["GET","POST"])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            reply = {
                "success" : False,
                "message" : "Berkas tidak terkirim",
                "class" : "Z"
            }
            return jsonify(reply)
        file = request.files['file']
        if file.filename == '':
            reply = {
                "success" : False,
                "message" : "Berkas tidak terkirim",
                "class" : "Z"
            }
            return jsonify(reply)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename) #TODO: Hash filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Receive uploaded file
            file.save(file_path)

            # Classify_image and return JSON
            grades = ['A', 'B', 'C']
            image = cv2.imread(file_path)
            print("Image shape: " + str(image.shape))
            processed_image = normalize_image(image, 2)

            result = predict_image(processed_image, 'bw', grades, 1)
            result_grade = grades[int(result)]

            # Cleanup
            os.remove(file_path)

            reply = {
                "success" : True,
                "message" : "",
                "class" : result_grade
            }
            return jsonify(reply)
        else:
            reply = {
                "success" : False,
                "message" : "Berkas tidak diperbolehkan",
                "class" : "Z"
            }
            return jsonify(reply)

    return '''
        <html>
        <head>
            <title> Classifier </title>
        </head>
        <body>
            <h1> Whoops! </h1>result_grade
            <p> You've seemed to access the wrong URL </p>
            <form method="post" enctype="multipart/form-data">
                <input type=file name=file>
                <input type=submit value=Upload>
            </form>
        </body>
        </html>
    '''

if __name__ == "__main__":
    app.run()
