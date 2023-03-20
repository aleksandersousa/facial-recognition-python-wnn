import pytest
from src.wisard.discriminator import Discriminator


def test_create_discriminator():
    with pytest.raises(Exception, match='the bit_str_size must be divisible by tuple_size.'):
        Discriminator('s1', 8, 3)


def test_create_nodes():
    discriminator = Discriminator('s1', 8, 2)
    assert len(discriminator.get_nodes()) == 4


def test_basic_train():
    discriminator = Discriminator('s1', 8, 2)
    discriminator.train('11010100')

    index = int('11', 2)
    first_node = discriminator.get_nodes()[0]
    first_memory = first_node.get_memories()[index]

    index = int('01', 2)
    second_node = discriminator.get_nodes()[1]
    second_memory = second_node.get_memories()[index]

    assert first_memory.get_content() == 1
    assert second_memory.get_content() == 1


def test_bleaching_train():
    discriminator = Discriminator('s1', 8, 2)
    discriminator.train(pattern='11010100', bleaching=True)
    discriminator.train(pattern='11010100', bleaching=True)

    index = int('11', 2)
    node = discriminator.get_nodes()[0]
    memory = node.get_memories()[index]

    assert memory.get_content() == 2


def test_basic_test():
    discriminator = Discriminator('s1', 8, 2)
    discriminator.train('11010100')

    assert discriminator.test('11010100') == 4


def test_simple_bleaching_test():
    discriminator = Discriminator('s1', 8, 2)
    discriminator.train('11010100', bleaching=True)

    assert discriminator.test('11010100', bleaching=True) == 4


def test_percentual_bleaching_test():
    discriminator = Discriminator('s1', 8, 2)

    discriminator.train('11010100', bleaching=True)
    discriminator.update_num_of_disc_patterns(1)

    assert discriminator.test(
        '11010100', bleaching=True, bleaching_type='percentual') == 4
