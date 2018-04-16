import cv2
import glob
import os

def normalize_image(image, toCanny=False):
    if canny:
        # TODO: Optimize 2nd and 3rd argument (minVal and maxVal)
        image = cv2.Canny(image, 100, 200)
    else:
        image = cv2.cvtColor(beras, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(beras, (350, 350))
    return image

def prepare_directory(grades):
    print('Preparing directory')
    data_dir = '../data'
    if not os.path.exists(data_dir):
        print('Creating data directory')
        os.makedirs(data_dir)

    for grade in grades:
        grade_data_dir = data_dir + '/{}'.format(grade)
        if not os.path.exists(grade_data_dir):
            print('Creating data/{} directory'.format(grade))
            os.makedirs(grade_data_dir)
        else:
            if os.listdir(grade_data_dir) != []:
                print('Directory {} is not empty, removing old data')
                filelist = glob.glob(grade_data_dir + '/*')
                for file in filelist:
                    os.remove(file)
    print("Finished preparing directory")

def process_raw_images(raw_images_path, grades, toCanny=False):
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
                cv2.imwrite('../data/{}/{}/{}'.format(img_type, grade, file_number + 1), processed_image)
            except:
                print("Error in processing {}".format(image))
                counter_failed += 1

    print("Image processing ({}) finished".format(img_type))
    print("Processed {} files with {} errors".format(counter_all, counter_failed))

if __name__ == '__main__':
    grades = ['A', 'B', 'C']

    yes = {'yes', 'ye', 'y', ''}
    no = {'no', 'n'}
    choice = input('Activate canny mode?[Y/n]').lower()

    while choice not in yes or choice not in no:
        print('Sorry, did not quite catch that')
        choice = input('Activate canny mode?[Y/n]').lower()

    canny_mode = False
    if choice in yes:
        canny_mode = True
        print("Canny mode activated")
    else:
        print("Black and White mode activated")

    path = input('Dataset path: ')
    if os.path.isdir(path):
        prepare_directory(grades)
        process_raw_images(path, grades, canny_mode)
    else:
        print('Could not find directory, exiting program')

