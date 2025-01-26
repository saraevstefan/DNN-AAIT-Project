from .dataset import DatasetAggregator, STSDataset, SimilarityDataset
from .read_biblical_01_dataset import Biblical_01_Dataset
from .read_paraphrase_ro_dataset import Paraphrase_RO_Dataset
from .read_ro_sts_dataset import RO_STS_Dataset


def load_dataset(dataset_name):
    if isinstance(dataset_name, list):
        return DatasetAggregator(
            [load_dataset(_dataset_name) for _dataset_name in dataset_name]
        )
    if dataset_name == "ro-sts":
        return RO_STS_Dataset("datasets/RO-STS")
    if dataset_name == "biblical_01":
        return Biblical_01_Dataset("datasets/Biblical_01")
    if dataset_name == "paraphrase-ro":
        return Paraphrase_RO_Dataset("datasets/Paraphrase-RO")

    raise ValueError(f"Unknown dataset {dataset_name}")


# Example Usage
# ro_sts = load_dataset("ro-sts")
# train_data = ro_sts.get_data("train")

# biblical_01 = load_dataset("biblical_01")
# train_data = biblical_01.get_data("train")
