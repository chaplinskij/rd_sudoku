import cv2, time
from abc import ABC, abstractmethod
from multiprocessing import Queue, Process

from services.image_service import ImageService
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
        if image_context.result is not None:
            image_context = ImageService.draw_context(image_context)

        return image_context

    def compute_result(self, image_context: ImageContext) -> ImageContext:
        if not self.input_queue.full():
            self.input_queue.put(image_context.image)

        if not self.output_queue.empty():
            image_context.result = self.output_queue.get()

        return image_context

    def queue_worker(self):
        """Метод для обработки кадра в процессе"""
        counter = 0
        while True:
            frame = self.input_queue.get()
            if frame is None:
                break
            # Симуляция длительной обработки

            time.sleep(1)  # Задержка для эмуляции сложной обработки
            counter += 1
            self.output_queue.put(counter)
