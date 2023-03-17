import cv2
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates


class ImageFile:
    def __init__(self):
        # INITIALIZING OBJECTS
        self.mp_face_detection = mp.solutions.face_detection

        # Setup the face detection function.
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5)

        # Initialize the mediapipe drawing class.
        self.mp_drawing = mp.solutions.drawing_utils

    def read_img(self, path):
        img = cv2.imread(path, 0)

        assert img is not None, "file could not be read, check with os.path.exists()"

        return img

    def get_img(self, path):
        return self.read_img(path) or None
