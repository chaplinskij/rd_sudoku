import numpy as np
import tensorflow as tf
import cv2


class DigitRecognizer:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        return tf.keras.models.load_model(model_path)

    def recognize_digits(self, images):
        results = []
        for img in images:
            processed_img = self.preprocess_image(img)
            prediction = self.model.predict(processed_img)
            digit = self.postprocess_prediction(prediction)
            results.append(digit)
        return results

    def preprocess_image(self, img):
        height, width, _ = img.shape
        img = img[3:height - 3, 3:width - 3]
        img_resized = cv2.resize(img, (28, 28))
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        img_normalized = img_gray / 255.0
        return np.expand_dims(img_normalized, axis=(0, -1))  # Добавляем размерности для модели

    def postprocess_prediction(self, prediction):
        digit = np.argmax(prediction)
        if np.max(prediction) < 0.5:  # Условие для неуверенности модели
            return 0
        return digit