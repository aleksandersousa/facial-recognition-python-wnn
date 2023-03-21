import pathlib
from src.file.dataset_file import DatasetFile


def test_read_training_dataset():
    file_path = str(pathlib.Path().absolute()) + \
        '/dataset/training/training_face_a_sp.csv'

    dataset_file = DatasetFile()
    data = dataset_file.read_dataset_file(file_path)

    assert data is not None
