import os
import numpy as np
import tensorflow as tf
from PIL import Image


class TrainService:
    pass

class Loader:
    @classmethod
    def load_digit_dataset(cls, data_dir='out', image_size=(28, 28), test_split=0.2):
        """
        Загружает датасет из указанного каталога в формате TensorFlow.
        """
        images = []
        labels = []

        for digit in range(10):
            digit_dir = os.path.join(data_dir, str(digit))
            if not os.path.exists(digit_dir):
                raise ValueError(f"Каталог с цифрой {digit} не найден: {digit_dir}")

            for file_name in os.listdir(digit_dir):
                file_path = os.path.join(digit_dir, file_name)
                img = Image.open(file_path).convert('L')  # Грузим изображение в градациях серого
                img = img.resize(image_size)  # Убедимся, что размер правильный
                images.append(img)
                labels.append(digit)

        # Преобразуем в numpy массивы
        images = np.array(images).reshape(-1, *image_size)  # Добавляем канал
        labels = np.array(labels)

        # Перемешиваем данные вручную
        indices = np.arange(len(images))
        np.random.shuffle(indices)
        images = images[indices]
        labels = labels[indices]

        # Разделяем данные
        split_index = int(len(images) * (1 - test_split))
        x_train, x_test = images[:split_index], images[split_index:]
        y_train, y_test = labels[:split_index], labels[split_index:]

        return x_train, x_test, y_train, y_test
