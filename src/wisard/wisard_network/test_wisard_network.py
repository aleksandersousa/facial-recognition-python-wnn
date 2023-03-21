from wisard.wisard_network.wisard_network import WisardNetwork


def test_unique():
    labels = ['s1', 's1', 's2', 's2']

    network = WisardNetwork(tuple_size=2)

    assert network._WisardNetwork__unique(labels) == ['s1', 's2']


def test_create_discriminators():
    training_labels = ['s1', 's1', 's2', 's2']
    training_input_patterns = ['10110010', '1010011', '01001101', '01011001']

    network = WisardNetwork(tuple_size=2)
    network.train(training_labels, training_input_patterns)

    assert len(network.get_discriminators()) == 2


def test_basic_train():
    training_labels = ['s1', 's1', 's2', 's2']
    training_input_patterns = ['10110010', '1010011', '01001101', '01011001']

    network = WisardNetwork(tuple_size=2)
    assert network.train(training_labels, training_input_patterns) == None


def test_bleaching_train():
    training_labels = ['s1', 's1', 's2', 's2']
    training_input_patterns = ['10110010', '1010011', '01001101', '01011001']

    network = WisardNetwork(tuple_size=2)
    assert network.train(
        training_labels, training_input_patterns, True) == None


def test_basic_test():
    training_labels = ['s1', 's1', 's2', 's2']
    training_input_patterns = ['10110010', '10100110', '01001101', '01001001']

    testing_labels = ['s1', 's1', 's2', 's2']
    testing_input_patterns = ['10110010', '10100110', '01001101', '01001001']

    network = WisardNetwork(tuple_size=2)
    network.train(training_labels, training_input_patterns)

    stats = network.test(testing_labels, testing_input_patterns)

    assert stats['hits_number'] == 4
    assert stats['tested_patterns_number'] == 4
    assert stats['hits_percentual'] == 100


def test_simple_bleaching_test():
    training_labels = ['s1', 's1', 's2', 's2']
    training_input_patterns = ['10110010', '10100110', '01001101', '01001001']

    testing_labels = ['s1', 's1', 's2', 's2']
    testing_input_patterns = ['10110010', '10100110', '01001101', '01001001']

    network = WisardNetwork(tuple_size=2)
    network.train(training_labels, training_input_patterns)

    stats = network.test(
        testing_labels, testing_input_patterns, bleaching=True)

    assert stats['hits_number'] == 4
    assert stats['tested_patterns_number'] == 4
    assert stats['hits_percentual'] == 100


def test_percentual_bleaching_test():
    training_labels = ['s1', 's1', 's2', 's2']
    training_input_patterns = ['10110010', '10100110', '01001101', '01001001']

    testing_labels = ['s1', 's1', 's2', 's2']
    testing_input_patterns = ['10110010', '10100110', '01001101', '01001001']

    network = WisardNetwork(tuple_size=2)
    network.train(training_labels, training_input_patterns)

    stats = network.test(
        testing_labels, testing_input_patterns, bleaching=True, bleaching_type='percentual')

    assert stats['hits_number'] == 4
    assert stats['tested_patterns_number'] == 4
    assert stats['hits_percentual'] == 100
