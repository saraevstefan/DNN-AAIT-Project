from .read_biblical_dataset import Biblical_Dataset
from .read_ro_sts_dataset import RO_STS_Dataset


def load_dataset(dataset_name):
    if dataset_name == "ro-sts":
        return RO_STS_Dataset("datasets/RO-STS")
    if dataset_name == "biblical":
        return Biblical_Dataset("datasets/Biblical")
    raise ValueError(f"Unknown dataset {dataset_name}")


# Example Usage
# ro_sts = load_dataset("ro-sts")
# train_data = ro_sts.get_data("train")

# biblical = load_dataset("biblical")
# train_data = biblical.get_data("train")
