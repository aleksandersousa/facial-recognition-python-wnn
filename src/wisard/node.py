from src.wisard.memory import Memory


class Node():
    def __init__(self, tuple_size) -> None:
        self.__tuple_size: int = tuple_size
        self.__memories: list = []
        self.__threshold: int = 0
        self.__num_of_disc_patterns: int = 0
        self.__greater_num_of_disc_patterns: int = 0

        self.__init_memory()

    def train(self, addr: str, bleaching=False) -> None:
        if bleaching:
            self.__bleaching_train(addr)
            return

        self.__basic_train(addr)

    def test(self, addr: str, bleaching=False, bleaching_type='simple') -> int:
        if bleaching:
            if bleaching_type == 'simple':
                return self.__simple_bleaching_test(addr)
            if bleaching_type == 'percentual':
                return self.__percentual_bleaching_test(addr)

        return self.__basic_test(addr)

    def inc_threshold(self) -> None:
        self.__threshold += 1

    def dec_threshold(self) -> None:
        if self.__threshold > 0:
            self.__threshold -= 1

    def reset_threshold(self) -> None:
        self.__threshold = 0

    # getters and setters

    def get_threshold(self) -> int:
        return self.__threshold

    def get_memories(self) -> list:
        return self.__memories

    def get_num_of_disc_patterns(self) -> int:
        return self.__num_of_disc_patterns

    def get_greater_num_of_disc_patterns(self) -> int:
        return self.__greater_num_of_disc_patterns

    def set_num_of_disc_patterns(self, num_of_disc_patterns: int) -> None:
        self.__num_of_disc_patterns = num_of_disc_patterns

    def set_greater_num_of_disc_patterns(self, greater_num_of_disc_patterns: int) -> None:
        self.__greater_num_of_disc_patterns = greater_num_of_disc_patterns

    # private

    def __basic_train(self, addr: str) -> None:
        index = int(addr, 2)
        self.__memories[index].active()

    def __bleaching_train(self, addr: str) -> None:
        index = int(addr, 2)
        self.__memories[index].inc_num_of_acc()

    def __basic_test(self, addr: str) -> int:
        index = int(addr, 2)
        return self.__memories[index].get_content()

    def __simple_bleaching_test(self, addr: str) -> int:
        index = int(addr, 2)
        return 1 if self.__memories[index].get_content() > self.__threshold else 0

    def __percentual_bleaching_test(self, addr: str) -> int:
        index = int(addr, 2)

        x = self.__memories[index].get_content() / self.__num_of_disc_patterns
        y = self.__threshold / self.__greater_num_of_disc_patterns

        return 1 if x > y else 0

    def __init_memory(self) -> None:
        num_of_addrs = 2 ** self.__tuple_size
        rotation = num_of_addrs // 2
        addresses = ['' for _ in range(num_of_addrs)]

        while rotation != 0:
            counter = 0
            value = 0

            for i in range(num_of_addrs):
                if counter >= rotation:
                    counter = 0
                    value = (value + 1) % 2

                addresses[i] += str(value)
                counter += 1

            rotation = int(rotation / 2)

        for address in addresses:
            memory = Memory(address)
            self.__memories.append(memory)
