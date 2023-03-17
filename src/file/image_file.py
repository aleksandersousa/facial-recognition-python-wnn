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
        return cv2.imread(path, 0)

    def get_img_input(self, img):
        image_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return image_input

    def get_rect_points(self, img, img_input):
        image_rows, image_cols = img.shape

        results = self.face_detection.process(img_input)
        detection = results.detections[0]
        location = detection.location_data

        relative_bounding_box = location.relative_bounding_box

        rect_start_point = _normalized_to_pixel_coordinates(
            relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
            image_rows)
        rect_end_point = _normalized_to_pixel_coordinates(
            relative_bounding_box.xmin + relative_bounding_box.width,
            relative_bounding_box.ymin + relative_bounding_box.height, image_cols,
            image_rows)

        return (rect_start_point, rect_end_point)

    def crop_img(self, rect_points, img_input):
        xleft, ytop = rect_points[0]
        xright, ybot = rect_points[1]

        crop_img = img_input[ytop: ybot, xleft: xright]
        return crop_img
