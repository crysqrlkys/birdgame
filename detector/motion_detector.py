import cv2
import numpy as np

from detector.base import AbstractDetector
from detector.utils import get_timer_coords


class MotionDetector(AbstractDetector):

    MIN_AREA = 80
    MAX_AREA = 50000

    def __init__(self, monitor_index: int | None = None, *args, **kwargs):
        super().__init__(monitor_index=monitor_index, *args, **kwargs)
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=300, detectShadows=False)
        self.timer_coords = get_timer_coords(self.monitor)

    def _apply_color_masks(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([0, 0, 0])
        upper_hsv = np.array([25, 255, 255])
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # hide timer
        x1, y1, x2, y2 = self.timer_coords
        mask[y1:y2, x1:x2] = 0

        return mask

    def _clean_mask_morphology(self, mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _extract_bounding_boxes(self, mask):
        bounding_boxes = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)

            if self.MIN_AREA < area < self.MAX_AREA:
                bounding_box = cv2.boundingRect(contour)
                bounding_boxes.append(bounding_box)

        return bounding_boxes

    def process_frame(self, frame):
        foreground_mask = self.fgbg.apply(frame)
        color_mask = self._apply_color_masks(frame)
        combined_mask = cv2.bitwise_and(foreground_mask, color_mask)

        mask = self._clean_mask_morphology(combined_mask)

        bounding_boxes = self._extract_bounding_boxes(mask)

        return bounding_boxes, mask

    def draw_boxes(self, frame, boxes):
        for x, y, w, h in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
            cv2.putText(
                frame,
                text="X",
                org=(x, y - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 0, 255),
                thickness=3,
            )

        return frame
