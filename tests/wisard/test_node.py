from src.wisard.node import Node


def test_init_memory():
    node = Node(3)

    addresses_out = ['000', '001', '010', '011', '100', '101', '110', '111']
    addresses_in = []

    for mem in node.get_memories():
        addresses_in.append(mem.get_address())

    assert addresses_in == addresses_out


def test_inc_threshold():
    node = Node(2)
    node.inc_threshold()

    assert node.get_threshold() == 1


def test_dec_threshold():
    node = Node(2)
    node.inc_threshold()
    node.dec_threshold()

    assert node.get_threshold() == 0


def test_reset_threshold():
    node = Node(2)
    node.reset_threshold()

    assert node.get_threshold() == 0


def test_basic_train():
    node = Node(2)
    addr = '00'

    node.train(addr)
    index = int(addr, 2)

    assert node.get_memories()[index].get_content() == 1


def test_bleaching_train():
    node = Node(2)
    addr = '00'

    node.train(addr, bleaching=True)
    node.train(addr, bleaching=True)
    index = int(addr, 2)

    assert node.get_memories()[index].get_content() == 2


def test_basic_test():
    node = Node(2)
    addr = '00'

    node.train(addr)

    assert node.test(addr) == 1


def test_simple_bleaching_test():
    node = Node(2)
    addr = '00'

    node.train(addr, bleaching=True)
    node.train(addr, bleaching=True)
    node.inc_threshold()

    assert node.test(addr, bleaching=True) == 1


def test_percentual_bleaching_test():
    node = Node(2)
    addr = '00'

    node.train(addr, bleaching=True)
    node.train(addr, bleaching=True)

    node.inc_threshold()

    node.set_num_of_disc_patterns(1)
    node.set_greater_num_of_disc_patterns(1)

    assert node.test(addr, bleaching=True, bleaching_type='percentual') == 1
