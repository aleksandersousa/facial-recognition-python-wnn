from image_processor.image_processor import ImageProcessor
from wisard.wisard_network import WisardNetwork


class Perform():
    def __init__(self) -> None:
        self.image_processor = ImageProcessor()

    def start(self):
        print('processing images...')
        print()

        self.__process_images()

        print()
        print('finished image processing!')

        print('executing network...')
        print()

        self.__execute()

        print()
        print('network finished!')

    # private

    def __execute(self):
        training_labels = ['s1', 's1', 's2', 's2']
        training_input_patterns = ['10110010',
                                   '10100110', '01001101', '01001001']

        testing_labels = ['s1', 's1', 's2', 's2']
        testing_input_patterns = ['10110010',
                                  '10100110', '01001101', '01001001']

        network = WisardNetwork(tuple_size=2)
        network.train(training_labels, training_input_patterns)
        network.test(testing_labels, testing_input_patterns)

    def __process_images(self) -> None:
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
