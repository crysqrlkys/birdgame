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

        # cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Trackbars", 800, 600)

        # cv2.createTrackbar("LH", "Trackbars", 0, 179, self._noop)
        # cv2.createTrackbar("LS", "Trackbars", 30, 255, self._noop)
        # cv2.createTrackbar("LV", "Trackbars", 0, 255, self._noop)
        # cv2.createTrackbar("UH", "Trackbars", 179, 179, self._noop)
        # cv2.createTrackbar("US", "Trackbars", 255, 255, self._noop)
        # cv2.createTrackbar("UV", "Trackbars", 255, 255, self._noop)

        # cv2.createTrackbar("LL", "Trackbars", 0, 255, self._noop)
        # cv2.createTrackbar("LGR", "Trackbars", 0, 255, self._noop)
        # cv2.createTrackbar("LBY", "Trackbars", 0, 255, self._noop)
        # cv2.createTrackbar("UL", "Trackbars", 255, 255, self._noop)
        # cv2.createTrackbar("UGR", "Trackbars", 255, 255, self._noop)
        # cv2.createTrackbar("UBY", "Trackbars", 255, 255, self._noop)

    # def _noop(self, _):
    #     pass

    def _apply_color_masks(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        # lh = cv2.getTrackbarPos("LH", "Trackbars")
        # ls = cv2.getTrackbarPos("LS", "Trackbars")
        # lv = cv2.getTrackbarPos("LV", "Trackbars")
        # uh = cv2.getTrackbarPos("UH", "Trackbars")
        # us = cv2.getTrackbarPos("US", "Trackbars")
        # uv = cv2.getTrackbarPos("UV", "Trackbars")

        lower_hsv = np.array([0, 0, 0])
        upper_hsv = np.array([25, 255, 255])
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # ll = cv2.getTrackbarPos("LL", "Trackbars")
        # lgr = cv2.getTrackbarPos("LGR", "Trackbars")
        # lby = cv2.getTrackbarPos("LBY", "Trackbars")
        # ul = cv2.getTrackbarPos("UL", "Trackbars")
        # ugr = cv2.getTrackbarPos("UGR", "Trackbars")
        # uby = cv2.getTrackbarPos("UBY", "Trackbars")

        # lower_lab = np.array([0, 0, 0])
        # upper_lab = np.array([255, 255, 255])
        # mask_lab = cv2.inRange(lab, lower_lab, upper_lab)

        # mask = cv2.bitwise_and(mask_hsv, mask_lab)
        mask = mask_hsv

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
