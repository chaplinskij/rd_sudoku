import cv2
import numpy as np

from services.context import ImageContext


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
    def split_into_cells(cls, warped_image):
        height, width = warped_image.shape[:2]
        cell_height, cell_width = height // 9, width // 9
        cells = []

        for row in range(9):
            for col in range(9):
                y1, y2 = row * cell_height, (row + 1) * cell_height
                x1, x2 = col * cell_width, (col + 1) * cell_width
                cell = warped_image[y1:y2, x1:x2]
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
        result_height, result_width = image_result.shape[:2]
        cv2.imshow("result", image_context.result)

        polygon = np.array(image_context.corners)
        src_points = np.array([[0, 0], [result_width, 0], [result_width, result_height], [0, result_height]], dtype=np.float32)
        dst_points = polygon.astype(np.float32)
        M = cv2.getPerspectiveTransform(src_points, dst_points)

        transformed_image_result = cv2.warpPerspective(image_result, M, (width, height))

        image_context.image = cv2.add(image, transformed_image_result)

        return image_context

    @classmethod
    def draw_digits(cls, image, image_context: ImageContext):
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
