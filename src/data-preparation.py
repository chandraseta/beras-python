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
    path = input('Dataset path: ')

    if os.path.isdir(path):
        process_raw_images(path, grades)

