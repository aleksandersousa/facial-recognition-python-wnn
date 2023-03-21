class Memory():
    def __init__(self, address) -> None:
        self.__address: str = address
        self.__content: int = 0

    def active(self) -> None:
        self.__content = 1

    def inc_num_of_acc(self) -> None:
        self.__content += 1

    def get_address(self) -> str:
        return self.__address

    def get_content(self) -> int:
        return self.__content
