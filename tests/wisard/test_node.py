from wisard.node import Node

addresses_out = ['00', '01', '10', '11']


def test_init_memory():
    node = Node(2)

    addresses_in = []

    for mem in node.get_memories():
        addresses_in.append(mem.get_address())

    assert addresses_in == addresses_out
