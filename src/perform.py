import pathlib

from image_processor.image_processor import ImageProcessor
from wisard.wisard_network.wisard_network import WisardNetwork
from file.dataset_file.dataset_file import DatasetFile


class Perform():
    def __init__(self) -> None:
        self.image_processor = ImageProcessor()
        self.dataset_file = DatasetFile()

        self.datasets = [
            {
                'training': str(pathlib.Path().absolute()) +
                '/dataset/training/training_face_a_ap.csv',
                'testing': str(pathlib.Path().absolute()) +
                '/dataset/testing/testing_face_a_ap.csv'
            },
            {
                'training': str(pathlib.Path().absolute()) +
                '/dataset/training/training_face_a_sp.csv',
                'testing': str(pathlib.Path().absolute()) +
                '/dataset/testing/testing_face_a_sp.csv'
            },
            {
                'training': str(pathlib.Path().absolute()) +
                '/dataset/training/training_face_b_ap.csv',
                'testing': str(pathlib.Path().absolute()) +
                '/dataset/testing/testing_face_b_ap.csv'
            },
            {
                'training': str(pathlib.Path().absolute()) +
                '/dataset/training/training_face_b_sp.csv',
                'testing': str(pathlib.Path().absolute()) +
                '/dataset/testing/testing_face_b_sp.csv'
            }
        ]

    def start(self):
        print('processando imagens...')
        print()

        self.__process_images()

        print()
        print('processamento de imagens finalizado!')
        print()

        print('executando a rede...')
        print()

        self.__execute()

        print()
        print('rede finalizada!')

    # private

    def __execute(self):
        DATASET_TYPE = 2

        training_labels = self.dataset_file.read_dataset_file(
            self.datasets[DATASET_TYPE]['training'])['labels']
        training_input_patterns = self.dataset_file.read_dataset_file(
            self.datasets[DATASET_TYPE]['training'])['bits']

        testing_labels = self.dataset_file.read_dataset_file(
            self.datasets[DATASET_TYPE]['testing'])['labels']
        testing_input_patterns = self.dataset_file.read_dataset_file(
            self.datasets[DATASET_TYPE]['testing'])['bits']

        network = WisardNetwork(tuple_size=4)
        network.train(labels=training_labels,
                      input_patterns=training_input_patterns)
        results = network.test(
            labels=testing_labels, input_patterns=testing_input_patterns)

        self.__print_results(results)

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

    def __print_results(self, results):
        print("Numero de padrões testados: {}".format(
            results['tested_patterns_number']))
        print()

        print("Numero de acertos: {}".format(
            results['hits_number']))
        print()

        print("Percentual de acerto: {}".format(
            results['hits_percentual']))
