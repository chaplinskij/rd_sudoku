import cv2, time
import numpy as np
from abc import ABC, abstractmethod
from multiprocessing import Queue, Process

from services.image_service import (
    ImageDrawService,
    ImageService,
)
from services.context import ImageContext


class BaseProcessor(ABC):
    def __init__(self):
        self.input_queue = Queue(maxsize=1)
        self.output_queue = Queue(maxsize=1)
        self.process = None

    def __enter__(self):
        """Автоматически запускает процесс при входе в контекст"""
        if not self.process:
            self.process = Process(target=self.queue_worker)
            self.process.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Автоматически завершает процесс при выходе из контекста"""
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join()
            self.process = None

    @abstractmethod
    def queue_worker(self):
        pass


class ImageProcessor(BaseProcessor):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings

    def process_frame(self, image_context: ImageContext) -> ImageContext:
        image_context.corners = ImageService.find_rectangle_corners(image_context)
        image_context = self.compute_result(image_context)
        image_context = ImageDrawService.draw_context(image_context)

        return image_context

    def compute_result(self, image_context: ImageContext) -> ImageContext:
        if not self.input_queue.full():
            self.input_queue.put(image_context)

        if not self.output_queue.empty():
            image_context.result = self.output_queue.get()

        return image_context

    def queue_worker(self):
        """Метод для обработки кадра в процессе"""
        while True:
            image_context = self.input_queue.get()
            if image_context.image is None or image_context.corners is None:
                continue
            # Симуляция длительной обработки
            transformed_image = ImageService.perspective_transform(image_context)


            # time.sleep(1)  # Задержка для эмуляции сложной обработки
            image_context.sudoku_result = [1, 2, 0, 0, 0, 0, 5, 8, 7, 4, 0, 0, 5, 0, 6, 0, 0, 8, 4, 0, 0, 9, 0, 0, 0, 4, 0, 0, 2, 8, 9, 1 , 3,
                      6, 0, 0 , 5, 8,2, 0, 0, 0, 0, 5, 8, 7, 4, 0, 0, 5, 0, 6, 0, 0, 8, 4, 0, 0, 9, 0, 8, 4, 0, 0, 3, 0,
                      0, 6, 0, 0, 8, 4, 0, 0, 0, 8, 4, 0, 0, 9, 0]

            result_image = np.zeros(transformed_image.shape, dtype=np.uint8)
            result_image = ImageDrawService.draw_digits(result_image, image_context)
            self.output_queue.put(result_image)
