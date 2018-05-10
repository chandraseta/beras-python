import cv2
import glob
import numpy as np
import os

def normalize_image(image, toCanny):

    image = cv2.resize(image, (500, 500))

    if toCanny:
        # Denoising
        image = cv2.fastNlMeansDenoisingColored(image, None, 25, 7, 9)

        # Add blur
        image = cv2.GaussianBlur(image, (3,3), 0)

        # Automatically calculate lower and upper bound
        sigma = 0.33

        grey = np.median(image)
        lower = int(max(0, (1.0 - sigma) * grey))
        upper = int(min(255, (1.0 + sigma) * grey))

        image = cv2.Canny(image, lower, upper)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    return image

def prepare_directory(grades, toCanny):
    img_types = ['bw', 'canny']
    print('Preparing directory')
    data_dir = '../data'
    if not os.path.exists(data_dir):
        print('Creating data directory')
        os.makedirs(data_dir)

    for img_type in img_types:
        type_dir = data_dir + '/{}'.format(img_type)
        if not os.path.exists(type_dir):
                print('Creating data/{} directory'.format(img_type))
                os.makedirs(type_dir)
        for grade in grades:
            grade_data_dir = type_dir + '/{}'.format(grade)
            if not os.path.exists(grade_data_dir):
                print('Creating data/{}/{} directory'.format(img_type, grade))
                os.makedirs(grade_data_dir)
            else:
                if (toCanny and img_type == 'canny') or (not toCanny and img_type == 'bw'):
                    if os.listdir(grade_data_dir) != []:
                        print('Directory {} is not empty, removing old data'.format(grade_data_dir))
                        filelist = glob.glob(grade_data_dir + '/*')
                        for file in filelist:
                            os.remove(file)
    print("Finished preparing directory")

def process_raw_images(raw_images_path, grades, toCanny):
    img_type = 'canny' if toCanny else 'bw'
    print('Processing raw beras data...')
    counter_all = 0
    counter_failed = 0
    for grade in grades:
        print('Processing data for class {}'.format(grade))
        images = glob.glob(raw_images_path + '/{}/*.jpg'.format(grade))
        for file_number, image in enumerate(images):
            frame = cv2.imread(image)
            processed_image = normalize_image(frame, toCanny)
            counter_all += 1
            try:
                cv2.imwrite('../data/{}/{}/{}.jpg'.format(img_type, grade, file_number + 1), processed_image)
                print("Data written to " + '../data/{}/{}/{}.jpg'.format(img_type, grade, file_number + 1))
            except:
                print("Error in processing {}".format(image))
                counter_failed += 1

    print("Image processing ({}) finished".format(img_type))
    print("Processed {} files with {} errors".format(counter_all, counter_failed))

if __name__ == '__main__':
    grades = ['A', 'B', 'C']

    yes = {'yes', 'ye', 'y', ''}
    no = {'no', 'n'}
    choice = input('Activate canny mode?[Y/n] ').lower()

    while choice not in yes and choice not in no:
        print('Sorry, did not quite catch that')
        choice = input('Activate canny mode?[Y/n] ').lower()

    canny_mode = False
    if choice in yes:
        canny_mode = True
        print("Canny mode activated")
    else:
        print("Black and White mode activated")

    path = input('Dataset path: ')
    if os.path.isdir(path):
        prepare_directory(grades, canny_mode)
        process_raw_images(path, grades, canny_mode)
    else:
        print('Could not find directory, exiting program')