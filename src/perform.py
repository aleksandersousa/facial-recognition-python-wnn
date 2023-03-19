from image_processor.image_processor import ImageProcessor


class Perform():
    def __init__(self) -> None:
        self.image_processor = ImageProcessor()

    def start(self):
        print('processing images...')
        print()

        self.process_images()

        print()
        print('finished image processing!')

    # private

    def process_images(self) -> None:
        self.image_processor.process_training_images(
            'face_a_ap', adaptative_threshold=True)
        self.image_processor.process_training_images(
            'face_a_sp', adaptative_threshold=True, shrink_pixels=True, shrink_size=4)
        self.image_processor.process_training_images('face_b_ap')
        self.image_processor.process_training_images(
            'face_b_sp', shrink_pixels=True, shrink_size=4)

        self.image_processor.process_testing_images(
            'face_a_ap', adaptative_threshold=True)
        self.image_processor.process_testing_images(
            'face_a_sp', adaptative_threshold=True, shrink_pixels=True, shrink_size=4)
        self.image_processor.process_testing_images('face_b_ap')
        self.image_processor.process_testing_images(
            'face_b_sp', shrink_pixels=True, shrink_size=4)
