import cv2
import numpy as np
from mss import mss

from detector.utils import get_monitor


class AbstractDetector:

    def __init__(self, monitor_index: int | None = None, *args, **kwargs):
        self.sct = mss()
        self.monitor = get_monitor(monitors=self.sct.monitors, index=monitor_index)

        cv2.namedWindow("Detect", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Detect", 1280, 720)

    def process_frame(self, frame, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    def draw_boxes(self, frame, bounding_boxes, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    def run(self, *args, **kwargs):
        while True:
            screenshot = np.array(self.sct.grab(self.monitor))
            frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

            bounding_boxes, mask = self.process_frame(frame)
            frame_with_boxes = self.draw_boxes(frame.copy(), bounding_boxes)

            cv2.imshow("Detect", frame_with_boxes)
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
