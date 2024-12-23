import cv2
import numpy as np
from abc import ABC, abstractmethod
from multiprocessing import Queue, Process
from time import time
from matplotlib import pyplot as plt

from services.image_service import (
    ImageDrawService,
    ImageService,
)
from services.color_balancer import ColorBalancer
from services.context import ImageContext
from services.recognize import DigitRecognizer
from services.sudoku import SudokuSolver


class BaseProcessor(ABC):
    def __init__(self):
        self.input_queue = Queue(maxsize=1)
        self.output_queue = Queue(maxsize=1)
        self.process = None
        self.log_messages = []

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

    def show_worker_status(self, message: str):
        print(message)

    def show_worker_status_window(self, message: str):
        self.log_messages.append(message)
        if len(self.log_messages) > 10:  # Ограничиваем лог 10 строками
            self.log_messages.pop(0)

        log_image = np.zeros((150, 250, 3), dtype=np.uint8)

        # Рендер текста
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)  # Белый текст
        line_thickness = 1
        y_offset = 20

        for i, log_message in enumerate(self.log_messages):
            y = y_offset + i * 20  # Расстояние между строками
            cv2.putText(log_image, log_message, (10, y), font, font_scale, font_color, line_thickness, cv2.LINE_AA)

        cv2.imshow('Log Window', log_image)


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
        self.show_worker_status('Worker: started')
        digit_recognizer = DigitRecognizer("data/simple_model.h5")
        while True:
            try:
                start_time = time()
                image_context = self.input_queue.get()
                if image_context.image is None or image_context.corners is None:
                    continue

                self.show_worker_status('Worker: 1. Processig image')
                transformed_image = ImageService.perspective_transform(image_context)
                cv2.imshow('transformed', transformed_image)

                # transformed_image = ColorBalancer.min_is_white(transformed_image)
                transformed_image = ImageService.image_correction(transformed_image)

                cells = ImageService.split_into_cells(transformed_image)
                sudoku_input = digit_recognizer.recognize_digits(cells)
                self.show_worker_status(f'Worker: recognized digits ({time() - start_time}) - {sudoku_input}')

                sudoku_result = SudokuSolver.solve(sudoku_input)
                if not sudoku_result:
                    self.show_worker_status(f'Worker: sudoku solution not founded ({time() - start_time})')
                    continue
                self.show_worker_status(f'Worker: Solve ({time() - start_time}) - {sudoku_result}')

                image_context.sudoku_result = sudoku_result
                result_image = np.zeros(transformed_image.shape, dtype=np.uint8)
                result_image = ImageDrawService.draw_digits(result_image, image_context)
                self.output_queue.put(result_image)

                self.show_worker_status(f'Worker: Finished at {time() - start_time}')
                self.show_worker_status('_'*50)
            except Exception as e:
                self.show_worker_status(f'Worker: get Exception - {e.with_traceback()}')
