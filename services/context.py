import cv2
import numpy as np


class ImageContext:
    def __init__(self, image = None, result = None, corners = None, result_size = None):
        self.image = image
        self.result = result
        self.corners = corners
        self.result_size = result_size or (324, 324) # TODO should passed from settings or constant as default
        self.digit_images = self.get_digit_images()
        self.sudoku_result = None

    def get_digit_images(self) -> dict:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 10
        thickness = 30
        margin_text = 30  # percentage
        result = {}
        for digit in range(1, 10):
            text = str(digit)
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            margin = int(text_height * margin_text / 100)
            text_width += margin + thickness // 2
            text_height += margin + thickness // 2
            text_image = np.zeros((text_height, text_width, 3), dtype=np.uint8)
            cv2.putText(text_image, text, (margin, text_height - margin // 2 - thickness // 2), font, font_scale,
                        (255, 0, 0), thickness, cv2.LINE_AA)

            result[digit] = {'image': text_image, 'shape': (text_height, text_width)}
        return result

