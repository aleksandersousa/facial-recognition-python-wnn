from src.wisard.node import Node


class Discriminator():
    def __init__(self, label: str, bit_str_size: int, tuple_size: int) -> None:
        if not bit_str_size % tuple_size == 0:
            raise Exception(
                'the bit_str_size must be divisible by tuple_size.')

        self.__label: str = label
        self.__tuple_size: int = tuple_size
        self.__pattern_num: int = 0
        self.__nodes: list = []

        num_of_nodes: int = bit_str_size // tuple_size
        self.__create_nodes(num_of_nodes)

    def train(self, pattern: str, bleaching=False) -> None:
        self.__pattern_num += 1

        start = 0
        end = self.__tuple_size

        for node in self.__nodes:
            addr = pattern[start:end]
            node.train(addr, bleaching)

            start += self.__tuple_size
            end += self.__tuple_size

    def test(self, pattern: str, bleaching=False, bleaching_type='simple') -> int:
        output = 0
        result = 0
        start = 0
        end = self.__tuple_size

        for node in self.__nodes:
            addr = pattern[start:end]
            result = node.test(addr, bleaching, bleaching_type)

            if result == 1:
                output += 1

            start += self.__tuple_size
            end += self.__tuple_size

        return output

    def inc_nodes_threshold(self) -> None:
        for node in self.__nodes:
            node.inc_threshold()

    def dec_nodes_threshold(self) -> None:
        for node in self.__nodes:
            node.dec_threshold()

    def reset_nodes_threshold(self) -> None:
        for node in self.__nodes:
            node.reset_threshold()

    def update_num_of_disc_patterns(self, greater_num_of_disc_patterns) -> None:
        for node in self.__nodes:
            node.set_num_of_disc_patterns(self.__pattern_num)
            node.set_greater_num_of_disc_patterns(greater_num_of_disc_patterns)

    # getters and setters

    def get_label(self) -> str:
        return self.__label

    def get_pattern_num(self) -> int:
        return self.__pattern_num

    def get_nodes(self) -> list:
        return self.__nodes

    # private

    def __create_nodes(self, num_of_nodes) -> None:
        for _ in range(num_of_nodes):
            node = Node(self.__tuple_size)
            self.__nodes.append(node)
