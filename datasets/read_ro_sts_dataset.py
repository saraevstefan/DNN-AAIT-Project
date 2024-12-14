import os

import pandas as pd

from .dataset import Dataset


class RO_STS_Dataset(Dataset):
    def _load_data(self):
        """
        Load the Ro-STS dataset from .tsv files.
        """
        for split in ["train", "dev", "test"]:
            file_path = os.path.join(self.data_dir, f"RO-STS.{split}.tsv")
            if os.path.exists(file_path):
                self.data[split] = pd.read_csv(
                    file_path, sep="\t", header=None, names=["score", "text1", "text2"]
                )
            else:
                raise FileNotFoundError(f"File {file_path} not found.")

    def _preprocess(self):
        """
        Example of preprocessing: Normalize scores.
        """
        for split, df in self.data.items():
            df["score"] = df["score"] / 5.0  # Normalize scores to [0, 1]

    def get_data(self, split):
        """
        Return the specified split of data (train, dev, or test).

        :param split: The dataset split to return ("train", "dev", or "test").
        :return: A pandas DataFrame containing the split data.
        """
        if split not in self.data:
            raise ValueError(
                f"Split '{split}' not found. Make sure to load the data first."
            )
        return self.data[split]
