import os

import pandas as pd

from .dataset import Dataset


class Paraphrase_RO_Dataset(Dataset):
    def _load_data(self):
        """
        Load the Paraphrase-RO dataset from .tsv files.
        """
        for split in ["train", "dev"]:
            file_path = os.path.join(self.data_dir, f"{split}.tsv")
            if os.path.exists(file_path):
                self.data[split] = pd.read_csv(
                    file_path, sep="\t", header=None, names=["text1", "text2"]
                )
            else:
                raise FileNotFoundError(f"File {file_path} not found.")

    def _preprocess(self):
        """
        Example of preprocessing: Normalize scores.
        """
        # do not edit the data
        return

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
