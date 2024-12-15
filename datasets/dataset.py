import pandas as pd
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self, data_dir):
        """
        Base class for dataset handling.

        :param data_dir: Path to the directory containing the dataset files.
        """
        self.data_dir = data_dir
        self.data = {}
        self._load_data()
        self._preprocess()

    def _load_data(self):
        """
        Load the dataset. This method should be implemented by subclasses.
        """
        raise NotImplementedError("load_data() must be implemented by subclasses.")

    def _preprocess(self):
        """
        Preprocess the dataset. This can include tokenization, normalization, etc.
        """
        raise NotImplementedError("preprocess() must be implemented by subclasses.")

    def get_data(self, split):
        """
        Return the data for training/testing.
        """
        raise NotImplementedError("get_data() must be implemented by subclasses.")

    def _train_test_split_rare(self, data, test_size, stratify_column):
        """
        Call train_test_split on the data while keeping
        the same distribution of values on column `stratify_column`
        """
        stratify_counts = data[stratify_column].value_counts()
        rare_entries = stratify_counts[stratify_counts == 1].index

        rare_data = data[data[stratify_column].isin(rare_entries)]
        stratifiable_data = data[~data[stratify_column].isin(rare_entries)]

        # Split the stratifiable data
        train, test = train_test_split(
            stratifiable_data,
            test_size=test_size,
            stratify=stratifiable_data[stratify_column],
            random_state=42,
        )

        test = pd.concat([test, rare_data])

        return train, test


class DatasetAggregator(Dataset):
    def __init__(self, datasets):
        self.__datasets = datasets

    def _load_data(self):
        return

    def _preprocess(self):
        return

    def get_data(self, split):
        return pd.concat([dataset.get_data(split) for dataset in self.__datasets])
