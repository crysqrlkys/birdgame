import sys

from detector.motion_detector import MotionDetector
from detector.retinanet_detector import RetinaNetDetector


class DetectorFactory:

    @staticmethod
    def create_detector(detector_type: str):

        detectors = {
            "--cv": MotionDetector,
            "--ai": RetinaNetDetector,
        }

        detector_class = detectors.get(detector_type)
        if detector_class:
            return detector_class()
        return None


def main():
    if len(sys.argv) < 2:
        print("Provide detector type --cv or --ai")
        sys.exit(1)

    detector_type = sys.argv[1]
    detector = DetectorFactory.create_detector(detector_type)

    if not detector:
        print(f"Error: Invalid detector type '{detector_type}' (--cv, --ai)")
        sys.exit(1)

    try:
        detector.run()
    except KeyboardInterrupt:
        print("Stopped by user")


if __name__ == "__main__":
    main()
