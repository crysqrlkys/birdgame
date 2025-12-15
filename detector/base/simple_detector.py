import cv2
import numpy as np

from detector.base.base import AbstractDetector


class SimpleDetector(AbstractDetector):

    MIN_AREA = 80
    MAX_AREA = 50000

    def run(self):
        while True:
            screenshot = np.array(self.sct.grab(self.monitor))
            frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

            boxes, mask = self.process_frame(frame)
            frame_with_boxes = self.draw_boxes(frame.copy(), boxes)

            cv2.imshow(self.window_name, frame_with_boxes)
            if mask is not None:
                cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Mask", 800, 600)
                cv2.imshow("Mask", mask)

            key = cv2.waitKey(1)
            if key & 0xFF == 27:
                break
            elif key & 0xFF == ord("q"):
                cv2.imwrite("screenshot.png", frame_with_boxes)

        cv2.destroyAllWindows()

    def draw_boxes(self, frame, boxes):
        for x, y, w, h in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
            cv2.putText(
                frame,
                text="Bird",
                org=(x, y - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 255, 0),
                thickness=3,
            )

        return frame
