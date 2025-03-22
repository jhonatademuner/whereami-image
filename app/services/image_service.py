import cv2
import numpy as np

def extract_features(file):
    np_image = np.frombuffer(file, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    histogram = get_histogram(gray_image)

    thumbnail = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    return histogram, thumbnail

def get_histogram(gray_image):
    histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    histogram = cv2.normalize(histogram, histogram).flatten()

    return histogram