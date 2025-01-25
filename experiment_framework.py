import json
import os
import random
from pprint import pprint

import pytorch_lightning as pl
import torch
from googletrans import Translator
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from architecture import MyDataset, TransformerModel
from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
translator = Translator()


def translate_to_random_language_and_back(lst_text):
    languages = ["en", "fr", "de", "es", "it"]
    try:
        # Randomly select a target language
        target_language = random.choice(languages)
        # Translate to the selected language
        translated = [
            res.text
            for res in translator.translate(lst_text, src="ro", dest=target_language)
        ]
        # Translate back to Romanian
        back_translated = [
            res.text
            for res in translator.translate(translated, src=target_language, dest="ro")
        ]
        return back_translated
    except Exception as e:
        print(f"Translation error: {e}")
        # Return the original text if translation fails
        return lst_text


def my_collate(model, batch, translate=False):
    # batch is a list of batch_size number of instances; each instance is a dict, as given by MyDataset.__getitem__()
    # return is a sentence1_batch, sentence2_batch, sims
    # the first two return values are dynamic batching for sentences 1 and 2, and [bs] is the sims for each of them
    # sentence1_batch is a dict like:
    """
    'input_ids': tensor([[101, 8667, 146, 112, 182, 170, 1423, 5650, 102],
                         [101, 1262, 1330, 5650, 102, 0, 0, 0, 0],
                         [101, 1262, 1103, 1304, 1304, 1314, 1141, 102, 0]]),
    'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0]]),
    'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1],
    """
    sentence1_batch = []
    sentence2_batch = []
    sims = []
    for instance in batch:
        # print(instance["sentence1"])
        sentence1_batch.append(instance["sentence1"])
        sentence2_batch.append(instance["sentence2"])
        sims.append(instance["sim"])

    if translate:
        if random.random() < 0.5:
            sentence1_batch = translate_to_random_language_and_back(sentence1_batch)
        else:
            sentence2_batch = translate_to_random_language_and_back(sentence2_batch)

    sentence1_batch = model.tokenizer(
        sentence1_batch,
        padding=True,
        max_length=model.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    sentence2_batch = model.tokenizer(
        sentence2_batch,
        padding=True,
        max_length=model.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    sims = torch.tensor(sims, dtype=torch.float)

    return sentence1_batch, sentence2_batch, sims


def prepare_model(args):
    # need to load for tokenizer
    # but also for training
    model = TransformerModel(
        model_name=args.model_name,
        lr=args.lr,
        model_max_length=args.model_max_length,
    )

    return model


def load_data(args):
    raw_train_dataset = load_dataset(args.train_dataset_name)
    raw_dev_dataset = load_dataset(args.dev_dataset_name)
    raw_test_dataset = load_dataset(args.test_dataset_name)

    train_dataset = MyDataset(data=raw_train_dataset.get_data("train"))
    dev_dataset = MyDataset(data=raw_dev_dataset.get_data("dev"))
    test_dataset = MyDataset(data=raw_test_dataset.get_data("test"))

    return train_dataset, dev_dataset, test_dataset


def prepare_data(args, model, datasets):
    train_dataset, dev_dataset, test_dataset = datasets

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        collate_fn=(
            lambda batch: my_collate(
                model, batch, translate=args.data_augmentation_translate_data
            )
        ),
        pin_memory=True,
    )
    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        collate_fn=(lambda batch: my_collate(model, batch)),
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        collate_fn=(lambda batch: my_collate(model, batch)),
        pin_memory=True,
    )

    print("Train dataset has {} instances.".format(len(train_dataset)))
    print("Dev dataset has {} instances.".format(len(dev_dataset)))
    print("Test dataset has {} instances.\n".format(len(test_dataset)))

    return train_dataloader, dev_dataloader, test_dataloader


def train_model(args, model, dataloaders, hyperparameters):
    train_dataloader, dev_dataloader, test_dataloader = dataloaders

    print(f"Running experiment with hyperparams {hyperparameters}")

    checkpoint_callback = ModelCheckpoint(
        monitor="dev/pearson",
        mode="max",
        save_top_k=1,
        dirpath="checkpoints/",
        filename="best_model",
    )

    early_stop = EarlyStopping(
        monitor="dev/pearson", patience=4, verbose=True, mode="max"
    )

    trainer = pl.Trainer(
        devices=args.gpus,
        callbacks=[early_stop, checkpoint_callback],
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
        enable_checkpointing=False,
        max_epochs=args.max_train_epochs,
    )

    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, train_dataloader, dev_dataloader)

    # Load the best model based on dev/pearson score
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        model = TransformerModel.load_from_checkpoint(
            best_model_path,
            model_name=args.model_name,
            lr=args.lr,
            model_max_length=args.model_max_length,
        )

    result_dev = trainer.test(model, dev_dataloader)
    result_test = trainer.test(model, test_dataloader)

    # Remove checkpoint after evaluation
    if best_model_path and os.path.exists(best_model_path):
        os.remove(best_model_path)

    print("Done, writing results...")
    result = {
        "dev_pearson": result_dev["test/pearson"],
        "dev_spearman": result_dev["test/spearman"],
        "dev_loss": result_dev["test/avg_loss"],
        "test_pearson": result_test["test/pearson"],
        "test_spearman": result_test["test/spearman"],
        "test_loss": result_test["test/avg_loss"],
    }

    results_location = os.path.join(trainer.logger.log_dir, "results.json")
    with open(results_location, "w") as f:
        json.dump(result, f, indent=4, sort_keys=True)

    return result


def get_experiments(grid_search):
    from itertools import product

    # Extract keys and values
    keys = grid_search.keys()
    values = grid_search.values()

    # Generate all combinations of parameters
    combinations = list(product(*values))

    # Convert combinations into list of experiment configurations
    experiments = [dict(zip(keys, combination)) for combination in combinations]
    return experiments


def run_experiment(experiment_config):
    args = Configuration(**experiment_config)
    print(
        "Batch size is {}, accumulate grad batches is {}, final batch_size is {}\n".format(
            args.batch_size,
            args.accumulate_grad_batches,
            args.batch_size * args.accumulate_grad_batches,
        )
    )

    print("Loading model...")
    model = prepare_model(args)

    print("Loading data...")
    datasets = load_data(args)
    dataloaders = prepare_data(args, model, datasets)

    result = train_model(args, model, dataloaders, hyperparameters=experiment_config)

    pprint(result)


class Configuration:
    gpus: int = 1
    batch_size: int = 16
    accumulate_grad_batches: int = 16
    lr: float = 2e-05
    model_max_length: int = 512
    max_train_epochs: int = 20
    model_name: str = "dumitrescustefan/bert-base-romanian-cased-v1"
    train_dataset_name: str = "ro-sts"
    dev_dataset_name: str = "ro-sts"
    test_dataset_name: str = "ro-sts"
    data_augmentation_translate_data: bool = False

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            self.__setattr__(key, value)


if __name__ == "__main__":
    GRID_SEARCH = {
        # "max_train_epochs": [20],  # remove when running for real
        "train_dataset_name": [
            # "ro-sts",
            "biblical_01",  # comment biblical_01 because it is huuuge :D
            ["ro-sts", "biblical_01"],
        ],
        "accumulate_grad_batches": [16],
        "model_name": [
            # "dumitrescustefan/bert-base-romanian-cased-v1",
            "dumitrescustefan/bert-base-romanian-uncased-v1",
            # "readerbench/RoBERT-small",
            "readerbench/RoBERT-base",
        ],
        "loss_function": ["MSE", "AnglE"],
        "data_augmentation_translate_data": [True],
    }

    # auto - to run on a cluster
    EXPERIEMNTS = get_experiments(GRID_SEARCH)

    # manual - to run on local
    EXPERIEMNTS = [
        {
            "model_name": "readerbench/RoBERT-base",
            "loss_function": "AnglE",
            "data_augmentation_translate_data": True,
        },
    ]

    for i, experiment in enumerate(EXPERIEMNTS):
        run_experiment(experiment)
