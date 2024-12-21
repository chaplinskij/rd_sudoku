import cv2

from services.image_processor import ImageProcessor
from services.context import ImageContext


class Runner:
    def __init__(self, settings):
        self.settings = settings

    def run(self):
        cap = cv2.VideoCapture(0)
        image_context = ImageContext()
        with ImageProcessor(self.settings) as image_processor:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                image_context.image = frame
                image_context = image_processor.process_frame(image_context)
                cv2.imshow("Processed Video", image_context.image)

                # Выход по клавише ESC
                if cv2.waitKey(1) == 27:
                    break

        cap.release()
        cv2.destroyAllWindows()