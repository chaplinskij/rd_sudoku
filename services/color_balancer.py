import cv2
import numpy as np


class ColorBalancer:
    @staticmethod
    def min_is_white(image):
        white = np.array([200, 200, 200])
        coeffs = 255.0 / white

        # Apply white balancing and generate balanced image
        balanced = np.zeros_like(image, dtype=np.float32)
        for channel in range(3):
            balanced[..., channel] = image[..., channel] * coeffs[channel]

        # White patching does not guarantee that the dynamic range is preserved, images must be clipped.
        balanced = balanced / 255
        balanced[balanced > 1] = 1
        return  cv2.convertScaleAbs(balanced, alpha=255.0)

    @staticmethod
    def gray_world(image):
        """
        Реализация алгоритма Gray World.
        """
        # Разделение каналов RGB
        b, g, r = cv2.split(image)

        # Среднее значение для каждого канала
        mean_r = np.mean(r)
        mean_g = np.mean(g)
        mean_b = np.mean(b)

        # Среднее значение по всем каналам
        mean_gray = (mean_r + mean_g + mean_b) / 3

        # Вычисление коэффициентов коррекции
        kr = mean_gray / mean_r
        kg = mean_gray / mean_g
        kb = mean_gray / mean_b

        # Применение коэффициентов коррекции
        r = cv2.multiply(r, kr)
        g = cv2.multiply(g, kg)
        b = cv2.multiply(b, kb)

        # Объединение каналов обратно
        balanced_image = cv2.merge((b, g, r))
        return cv2.convertScaleAbs(balanced_image)

    @staticmethod
    def scale_by_max(image):
        """
        Реализация алгоритма Scale-by-Max.
        """
        # Разделение каналов RGB
        b, g, r = cv2.split(image)

        # Максимальное значение для каждого канала
        max_r = np.max(r)
        max_g = np.max(g)
        max_b = np.max(b)

        # Общий максимальный уровень для нормализации
        max_value = max(max_r, max_g, max_b)

        # Масштабирование каналов к общему максимуму
        r = cv2.multiply(r, max_value / max_r)
        g = cv2.multiply(g, max_value / max_g)
        b = cv2.multiply(b, max_value / max_b)

        # Объединение каналов обратно
        balanced_image = cv2.merge((b, g, r))
        return cv2.convertScaleAbs(balanced_image)