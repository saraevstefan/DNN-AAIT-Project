import os

import pandas as pd

from .dataset import Dataset


class Biblical_01_Dataset(Dataset):
    def _load_data(self):
        """
        Load the Biblical dataset from a .csv file.
        """

        if not os.path.exists(os.path.join(self.data_dir, "biblical.train.csv")):
            self.__split_dataset()
        else:
            for split in ["train", "dev", "test"]:
                file_path = os.path.join(self.data_dir, f"biblical.{split}.csv")
                if os.path.exists(file_path):
                    self.data[split] = pd.read_csv(file_path)
                else:
                    raise FileNotFoundError(f"File {file_path} not found.")

    def __split_dataset(self):
        """
        Split the dataset into train, dev, and test sets while maintaining score distribution.
        """
        data = None

        file_path = os.path.join(self.data_dir, "bibl_paraphrase_full.csv")
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
        else:
            raise FileNotFoundError(f"File {file_path} not found.")

        if data is None:
            raise ValueError("Data must be loaded before splitting.")

        drop, keep = self._train_test_split_rare(
            data, test_size=0.1, stratify_column="score"
        )
        train, temp = self._train_test_split_rare(
            keep, test_size=0.3, stratify_column="score"
        )
        dev, test = self._train_test_split_rare(
            temp, test_size=0.5, stratify_column="score"
        )

        # Save the splits as files
        for split_name, split_data in zip(["train", "dev", "test"], [train, dev, test]):
            split_path = os.path.join(self.data_dir, f"biblical.{split_name}.csv")
            split_data.to_csv(split_path, index=False)
            self.data[split_name] = split_data

    def _preprocess(self):
        """
        Example preprocessing: Normalize scores.
        """
        for split, df in self.data.items():
            df: pd.DataFrame
            df["score"] = (
                df["score"].astype(float) / 100.0
            )  # Normalize scores to [0, 1]

    def get_data(self, split):
        """
        Return the specified split of data (train, dev, or test).

        :param split: The dataset split to return ("train", "dev", or "test").
        :return: A pandas DataFrame containing the split data.
        """
        if split not in self.data:
            raise ValueError(
                f"Split '{split}' not found. Make sure to split the dataset first."
            )
        return self.data[split]
