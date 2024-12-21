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
    def draw_context(cls, image_context: ImageContext):
        image_context.image = cv2.putText(
            image_context.image,
            f'Result: {image_context.result}',
            (10, 450), cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        return image_context

