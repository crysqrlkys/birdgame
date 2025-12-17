import cv2
import numpy as np

from detector.base.base import AbstractDetector


class AiDetector(AbstractDetector):

    DETECT_INTERVAL = None
    CONFIDENCE_THRESHOLD = 0.5

    def _continuos_detection(self):
        while True:
            screenshot = np.array(self.sct.grab(self.monitor))
            frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
            boxes, labels = self.process_frame(frame)

            frame_with_boxes = self.draw_boxes(frame.copy(), boxes, labels)
            cv2.imshow(self.window_name, frame_with_boxes)

            key = cv2.waitKey(1)
            if key & 0xFF == 27:
                break
            elif key & 0xFF == ord("q"):
                cv2.imwrite("screenshot.png", frame_with_boxes)

        cv2.destroyAllWindows()

    # test (worse and slower than continuos)
    def _interval_detection(self):

        multi_tracker = cv2.legacy.MultiTracker_create()
        frame_count = 0

        while True:
            screenshot = np.array(self.sct.grab(self.monitor))
            frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

            frame_count += 1

            if (
                frame_count % self.DETECT_INTERVAL == 0
                or len(multi_tracker.getObjects()) == 0
            ):
                multi_tracker = cv2.legacy.MultiTracker_create()

                detected_boxes, labels = self.process_frame(frame)
                for box in detected_boxes:
                    tracker = cv2.legacy.TrackerMedianFlow_create()
                    multi_tracker.add(tracker, frame, box)

            _, boxes = multi_tracker.update(frame)
            frame_with_boxes = self.draw_boxes(frame.copy(), boxes, labels)
            cv2.imshow(self.window_name, frame_with_boxes)

            key = cv2.waitKey(1)
            if key & 0xFF == 27:
                break
            elif key & 0xFF == ord("q"):
                cv2.imwrite("screenshot.png", frame_with_boxes)

        cv2.destroyAllWindows()

    def run(self):
        if self.DETECT_INTERVAL:
            self._interval_detection()
        else:
            self._continuos_detection()

    def draw_boxes(self, frame: np.ndarray, boxes, labels=None) -> np.ndarray:
        labels = labels if labels is not None else [0] * len(boxes)
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = (int(c) for c in box)
            color = (0, 0, 255) if label else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=2)
            cv2.putText(
                frame,
                text="Dead" if label else "Alive",
                org=(x1, y1 - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=color,
                thickness=3,
            )
        return frame
