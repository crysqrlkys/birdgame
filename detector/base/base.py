import cv2
from mss import mss

from detector.constants import DEFAULT_RESULT_WINDOW_NAME
from detector.utils import get_monitor


class AbstractDetector:

    def __init__(
        self, monitor_index: int | None = None, window_name: str = None, *args, **kwargs
    ):
        self.sct = mss()
        self.monitor = get_monitor(monitors=self.sct.monitors, index=monitor_index)
        self.window_name = window_name or DEFAULT_RESULT_WINDOW_NAME
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)

    def process_frame(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    def draw_boxes(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    def run(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")
