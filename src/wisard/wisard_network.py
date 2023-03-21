import random
from wisard.discriminator import Discriminator


class WisardNetwork():
    def __init__(self, tuple_size: int) -> None:
        self.__tuple_size: int = tuple_size
        self.__discriminators: list = []
        self.__disc_hash: dict = {}
        self.__stats: dict = {
            'hits_number': 0,
            'tested_patterns_number': 0,
            'hits_percentual': 0
        }

    def train(self, labels: list, input_patterns: list, bleaching=False):
        if len(labels) != len(input_patterns):
            raise Exception(
                'labels length must be equal input_patterns length.')

        self.__create_discriminators(labels, input_patterns, self.__tuple_size)

        for i, pattern in enumerate(input_patterns):
            label = labels[i]
            self.__map_disc(label)

            self.__disc_hash[label].train(pattern, bleaching)

    def test(self, labels: list, input_patterns: list, bleaching=False, bleaching_type='simple') -> dict:
        if len(labels) != len(input_patterns):
            raise Exception(
                'labels length must be equal input_patterns length.')

        self.__stats: dict = {
            'hits_number': 0,
            'tested_patterns_number': 0,
            'hits_percentual': 0
        }

        if bleaching:
            if bleaching_type == 'simple':
                self.__bleaching_test(labels, input_patterns)
            else:
                self.__bleaching_test(labels, input_patterns, 'percentual')
        else:
            self.__basic_test(labels, input_patterns)

        return self.__stats

    # getters and setters

    def get_discriminators(self) -> list:
        return self.__discriminators

    # private

    def __basic_test(self, labels: list, input_patterns: list):
        for i, pattern in enumerate(input_patterns):
            greatest_output = 0
            output = 0
            output_label = ''
            greatest_discriminators = []

            label = labels[i]

            for discriminator in self.__discriminators:
                output = discriminator.test(pattern)

                if output > greatest_output:
                    greatest_output = output
                    greatest_discriminators.clear()
                    greatest_discriminators.append(discriminator)
                elif output == greatest_output:
                    greatest_discriminators.append(discriminator)

            if len(greatest_discriminators) == 1:
                output_label = greatest_discriminators[0].get_label()
            else:
                output_label = random.choice(greatest_discriminators)

            self.__compute_stats(label, output_label)

    def __bleaching_test(self, labels: list, input_patterns: list, bleaching_type='simple'):
        for i, pattern in enumerate(input_patterns):
            greater_num_of_disc_patterns = 0
            if bleaching_type == 'percentual':
                greater_num_of_disc_patterns = max(
                    disc.get_pattern_num() for disc in self.__discriminators)

            for discriminator in self.__discriminators:
                discriminator.reset_nodes_threshold()

                if bleaching_type == 'percentual':
                    discriminator.update_num_of_disc_patterns(
                        greater_num_of_disc_patterns)

            lastCase = False
            output_label = ''
            output = 0
            greatest_output = 0
            greatest_discriminators = []

            label = labels[i]

            while True:
                greatest_discriminators.clear()

                for discriminator in self.__discriminators:
                    output = discriminator.test(pattern)

                    if output > greatest_output:
                        greatest_output = output
                        greatest_discriminators.clear()
                        greatest_discriminators.append(discriminator)
                    elif output == greatest_output:
                        greatest_discriminators.append(discriminator)

                if not lastCase:
                    if len(greatest_discriminators) == 1:
                        output_label = greatest_discriminators[0].get_label()
                        break

                    if not greatest_discriminators:
                        lastCase = True

                        for discriminator in self.__discriminators:
                            discriminator.dec_nodes_threshold()
                    else:
                        for discriminator in self.__discriminators:
                            discriminator.inc_nodes_threshold()
                else:
                    output_label = random.choice(greatest_discriminators)
                    break

            self.__compute_stats(label, output_label)

    def __map_disc(self, label) -> None:
        if label not in self.__disc_hash:
            discriminator = next(
                (x for x in self.__discriminators if x.get_label() == label), None)
            self.__disc_hash[label] = discriminator

    def __compute_stats(self, output, expected_output):
        self.__stats['tested_patterns_number'] += 1

        if output == expected_output:
            self.__stats['hits_number'] += 1

        hits_percentual = (self.__stats['hits_number'] * 100) / \
            self.__stats['tested_patterns_number']
        self.__stats['hits_percentual'] = hits_percentual

    def __create_discriminators(self, labels, input_patterns, tuple_size) -> None:
        bit_str_size = len(input_patterns[0])
        unique_labels = self.__unique(labels)

        for i in range(len(unique_labels)):
            label = unique_labels[i]
            discriminator = Discriminator(label, bit_str_size, tuple_size)
            self.__discriminators.append(discriminator)

    def __unique(self, arr) -> list:
        return list(dict.fromkeys(arr))
