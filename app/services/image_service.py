import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.applications.InceptionV3(include_top=False, pooling="avg")

def compare_images(img1_bytes, img2_bytes):

    img1 = pre_process_image(img1_bytes)
    img2 = pre_process_image(img2_bytes)

    hist1 = extract_histogram(img1)
    hist2 = extract_histogram(img2)

    cnn1 = extract_cnn_features(img1)
    cnn2 = extract_cnn_features(img2)

    mse_difference = extract_mse_difference(img1, img2)

    hist_similarity = compare_histograms(hist1, hist2)
    cnn_similarity = compare_cnn_features(cnn1, cnn2)
    mse_similarity = 1 - mse_difference

    final_similarity = (hist_similarity * 0.3 + cnn_similarity * 0.5 + mse_similarity * 0.2) * 100

    return hist_similarity, cnn_similarity, mse_similarity, final_similarity


def pre_process_image(img_bytes):
    np_image = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    return image


def extract_histogram(image):
    image_resized = cv2.resize(image, (216, 384)) 
    hist_r = cv2.calcHist([image_resized], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image_resized], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([image_resized], [2], None, [256], [0, 256])

    hist_r = cv2.normalize(hist_r, hist_r).flatten()
    hist_g = cv2.normalize(hist_g, hist_g).flatten()
    hist_b = cv2.normalize(hist_b, hist_b).flatten()

    return np.concatenate([hist_r, hist_g, hist_b])


def extract_cnn_features(image):
    image = cv2.resize(image, (299, 299)) / 255.0 
    image = np.expand_dims(image, axis=0) 
    return model.predict(image).flatten()


def extract_mse_difference(image1, image2):

    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    image1 = image1.astype(np.float32) / 255.0
    image2 = image2.astype(np.float32) / 255.0
    
    return np.mean((image1 - image2) ** 2)


def compare_histograms(hist1, hist2):
    similarity = cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_CORREL)
    return (similarity + 1) / 2  


def compare_cnn_features(feat1, feat2):
    return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))