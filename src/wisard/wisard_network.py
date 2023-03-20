from src.wisard.discriminator import Discriminator


class WisardNetwork():
    def __init__(self, labels: list, input_patterns: list, tuple_size: int) -> None:
        if not len(labels) != len(input_patterns):
            raise Exception(
                'labels length must be equal input_patterns length.')

        self.__labels: list = labels
        self.__discriminators: list = []

        bit_str_size = len(input_patterns[0])
        self.__create_discriminators(bit_str_size, tuple_size)

    def __create_discriminators(self, bit_str_size, tuple_size) -> None:
        for i in range(self.__labels):
            label = self.__labels[i]
            discriminator = Discriminator(label, bit_str_size, tuple_size)
            self.__discriminators.append(discriminator)
