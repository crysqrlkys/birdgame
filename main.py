import os

from dotenv import load_dotenv

from detector.motion_detector import MotionDetector

load_dotenv()

DEBUG = os.getenv("DEBUG", False)


if __name__ == "__main__":
    MotionDetector().run()
