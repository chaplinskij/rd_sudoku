import cv2
import numpy as np
from matplotlib import pyplot as plt

from services.context import ImageContext
from services.color_balancer import ColorBalancer


class ImageService:
    @classmethod
    def _preprocess_image(cls, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)
        return threshold

    @classmethod
    def _find_largest_quadrilateral(cls, threshold):
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest_contour, True)
        polygone = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)

        return polygone if len(polygone) == 4 else None

    @classmethod
    def find_rectangle_corners(cls, image_context: ImageContext):
        threshold = cls._preprocess_image(image_context.image)
        quadrilateral = cls._find_largest_quadrilateral(threshold)
        if quadrilateral is None:
            return
        points = sorted(np.vstack(quadrilateral).squeeze(), key=lambda x: (x[1], x[0]))
        if not points:
            return
        top_left, top_right = sorted(points[:2], key=lambda x: x[0])
        bottom_left, bottom_right = sorted(points[2:], key=lambda x: x[0])

        return top_left, top_right, bottom_right, bottom_left

    @classmethod
    def perspective_transform(cls, image_context: ImageContext):
        width, height = image_context.result_size
        src = np.array(image_context.corners, dtype=np.float32)
        destination = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        matrix = cv2.getPerspectiveTransform(np.array(src, dtype="float32"), destination)
        warped = cv2.warpPerspective(image_context.image, matrix, (width, height))
        return warped

    @classmethod
    def image_correction(cls, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
        # _, black_and_white = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
        # image_bright = cv2.cvtColor(image_bright, cv2.COLOR_GRAY2BGR)
        # equalized = cv2.equalizeHist(gray)
        _, otsu_threshold = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 2)

        return cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)

    @classmethod
    def image_correction_2(cls, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image, (5, 5), 0)
        # Шаг 1: Повышение контраста с помощью CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(image)

        # Шаг 2: Удаление шума с помощью медианного размытия
        blurred_image = cv2.medianBlur(enhanced_image, 3)

        # Шаг 3: Бинаризация изображения
        binary_image = cv2.adaptiveThreshold(
            blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Шаг 4: Обнаружение контуров (опционально, для исправления геометрии)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(output_image, contours, -1, (0, 0, 0), 1)
        return output_image

    @classmethod
    def split_into_cells(cls, warped_image):
        height, width = warped_image.shape[:2]
        cell_height, cell_width = height // 9, width // 9
        cells = []

        for row in range(9):
            for col in range(9):
                y1, y2 = row * cell_height, (row + 1) * cell_height
                x1, x2 = col * cell_width, (col + 1) * cell_width
                cell = warped_image[y1:y2, x1:x2]
                # cell = ColorBalancer.gray_world(cell)
                cells.append(cell)

        return cells

class ImageDrawService:

    @classmethod
    def draw_context(cls, image_context: ImageContext) -> ImageContext:
        height, width = image_context.image.shape[:2]
        if image_context.corners is None or image_context.result is None:
            return image_context

        image = image_context.image
        image_result = image_context.result
        image_result = cv2.cvtColor(image_result, cv2.COLOR_RGB2BGR)
        result_height, result_width = image_result.shape[:2]
        cv2.imshow("result", image_result)

        polygon = np.array(image_context.corners)
        src_points = np.array([[0, 0], [result_width, 0], [result_width, result_height], [0, result_height]], dtype=np.float32)
        dst_points = polygon.astype(np.float32)
        M = cv2.getPerspectiveTransform(src_points, dst_points)

        transformed_image_result = cv2.warpPerspective(image_result, M, (width, height))

        image_context.image = cv2.add(image, transformed_image_result)

        return image_context

    @classmethod
    def draw_digits(cls, image, image_context: ImageContext):
        if not image_context.sudoku_result:
            return image
        height, width = image.shape[:2]
        cell_height, cell_width = height // 9, width // 9

        for col in range(9):
            for row in range(9):
                id = col + row*9
                y1, y2 = row * cell_height, (row + 1) * cell_height
                x1, x2 = col * cell_width, (col + 1) * cell_width

                digit = image_context.sudoku_result[id]
                if digit:
                    digit_image = image_context.digit_images[digit]['image']
                    digit_image = cv2.resize(digit_image, (cell_width, cell_height))
                    image[y1:y2, x1:x2] = digit_image
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 1)

        return image
