import pandas as pd


class DatasetFile():
    def read_dataset_file(self, file_path):
        data = pd.read_csv(file_path)

        return {
            'labels': data['label'].values,
            'bits': data['input_bits'].values
        }
