from memory import Memory


class Node():
    def __init__(self, tuple_size) -> None:
        self.__tuple_size: int = tuple_size
        self.__memories: list = []

        self.__init_memory()

    def get_memories(self) -> list:
        return self.__memories

    # private

    def __init_memory(self) -> None:
        num_of_addrs = self.__tuple_size ** 2
        rotation = int(num_of_addrs / 2)
        addresses = ['' for _ in range(num_of_addrs)]

        while rotation != 0:
            counter = 0
            value = 0

            for i in range(num_of_addrs):
                if counter >= rotation:
                    counter = 0
                    value = (value + 1) % 2

                addresses[i] += value
                counter += 1

            rotation = int(rotation / 2)

        for address in addresses:
            memory = Memory(address)
            self.__memories.append(memory)
