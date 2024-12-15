import json
import os
from pprint import pprint

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from scipy.stats import spearmanr
from scipy.stats.stats import pearsonr
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
)

from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TransformerModel(pl.LightningModule):
    def __init__(
        self,
        model_name="dumitrescustefan/bert-base-romanian-cased-v1",
        lr=2e-05,
        model_max_length=512,
        loss_function="cosine_similarity",
    ):
        super().__init__()
        print("Loading AutoModel [{}]...".format(model_name))
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(
            model_name, num_labels=1, output_hidden_states=True
        )
        self.model = AutoModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(0.2)

        self.loss_fct = MSELoss()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.lr = lr
        self.model_max_length = model_max_length

        self.train_y_hat = []
        self.train_y = []
        self.train_loss = []
        self.dev_y_hat = []
        self.dev_y = []
        self.dev_loss = []
        self.test_y_hat = []
        self.test_y = []
        self.test_loss = []

        # add pad token
        self.validate_pad_token()

    def validate_pad_token(self):
        if self.tokenizer.pad_token is not None:
            return
        if self.tokenizer.sep_token is not None:
            print(
                f"\tNo PAD token detected, automatically assigning the SEP token as PAD."
            )
            self.tokenizer.pad_token = self.tokenizer.sep_token
            return
        if self.tokenizer.eos_token is not None:
            print(
                f"\tNo PAD token detected, automatically assigning the EOS token as PAD."
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            return
        if self.tokenizer.bos_token is not None:
            print(
                f"\tNo PAD token detected, automatically assigning the BOS token as PAD."
            )
            self.tokenizer.pad_token = self.tokenizer.bos_token
            return
        if self.tokenizer.cls_token is not None:
            print(
                f"\tNo PAD token detected, automatically assigning the CLS token as PAD."
            )
            self.tokenizer.pad_token = self.tokenizer.cls_token
            return
        raise Exception(
            "Could not detect SEP/EOS/BOS/CLS tokens, and thus could not assign a PAD token which is required."
        )

    def forward(self, s1, s2, sim):
        o1 = self.model(
            input_ids=s1["input_ids"].to(self.device),
            attention_mask=s1["attention_mask"].to(self.device),
            return_dict=True,
        )
        o2 = self.model(
            input_ids=s2["input_ids"].to(self.device),
            attention_mask=s2["attention_mask"].to(self.device),
            return_dict=True,
        )
        pooled_sentence1 = o1.last_hidden_state  # [batch_size, seq_len, hidden_size]
        pooled_sentence1 = torch.mean(
            pooled_sentence1, dim=1
        )  # [batch_size, hidden_size]
        pooled_sentence2 = o2.last_hidden_state  # [batch_size, seq_len, hidden_size]
        pooled_sentence2 = torch.mean(
            pooled_sentence2, dim=1
        )  # [batch_size, hidden_size]

        cosines = self.cos(pooled_sentence1, pooled_sentence2).squeeze()  # [batch_size]
        loss = self.loss_fct(cosines, sim)
        return loss, cosines

    def training_step(self, batch, batch_idx):
        s1, s2, sim = batch

        loss, predicted_sims = self(s1, s2, sim)

        self.train_y_hat.extend(predicted_sims.detach().cpu().view(-1).numpy())
        self.train_y.extend(sim.detach().cpu().view(-1).numpy())
        self.train_loss.append(loss.detach().cpu().numpy())

        return {"loss": loss}

    def on_training_epoch_end(self):
        pearson_score = pearsonr(self.train_y, self.train_y_hat)[0]
        spearman_score = spearmanr(self.train_y, self.train_y_hat)[0]
        mean_train_loss = sum(self.train_loss) / len(self.train_loss)

        self.log("train/avg_loss", mean_train_loss, prog_bar=True)
        self.log("train/pearson", pearson_score, prog_bar=False)
        self.log("train/spearman", spearman_score, prog_bar=False)

        self.train_y_hat = []
        self.train_y = []
        self.train_loss = []

    def validation_step(self, batch, batch_idx):
        s1, s2, sim = batch

        loss, predicted_sims = self(s1, s2, sim)

        self.dev_y_hat.extend(predicted_sims.detach().cpu().view(-1).numpy())
        self.dev_y.extend(sim.detach().cpu().view(-1).numpy())
        self.dev_loss.append(loss.detach().cpu().numpy())

        return {"loss": loss}

    def on_validation_epoch_end(self):
        pearson_score = pearsonr(self.dev_y, self.dev_y_hat)[0]
        spearman_score = spearmanr(self.dev_y, self.dev_y_hat)[0]
        mean_dev_loss = sum(self.dev_loss) / len(self.dev_loss)

        self.log("dev/avg_loss", mean_dev_loss, prog_bar=True)
        self.log("dev/pearson", pearson_score, prog_bar=True)
        self.log("dev/spearman", spearman_score, prog_bar=True)

        self.dev_y_hat = []
        self.dev_y = []
        self.dev_loss = []

    def test_step(self, batch, batch_idx):
        s1, s2, sim = batch

        loss, predicted_sims = self(s1, s2, sim)

        self.test_y_hat.extend(predicted_sims.detach().cpu().view(-1).numpy())
        self.test_y.extend(sim.detach().cpu().view(-1).numpy())
        self.test_loss.append(loss.detach().cpu().numpy())

        return {"loss": loss}

    def on_test_epoch_end(self):
        pearson_score = pearsonr(self.test_y, self.test_y_hat)[0]
        spearman_score = spearmanr(self.test_y, self.test_y_hat)[0]
        mean_test_loss = sum(self.test_loss) / len(self.test_loss)

        self.log("test/avg_loss", mean_test_loss, prog_bar=True)
        self.log("test/pearson", pearson_score, prog_bar=True)
        self.log("test/spearman", spearman_score, prog_bar=True)

        self.test_y_hat = []
        self.test_y = []
        self.test_loss = []

    def configure_optimizers(self):
        return torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad], lr=self.lr, eps=1e-08
        )


class MyDataset(Dataset):
    def __init__(self, data):
        data_dict = data.to_dict("records")
        self.instances = [
            {
                "sentence1": entry["text1"],
                "sentence2": entry["text2"],
                "sim": entry["score"],
            }
            for entry in data_dict
        ]

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        return self.instances[i]  # torch.tensor([0], dtype=torch.long)


def my_collate(model, batch):
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
    raw_test_dataset = load_dataset(args.test_dataset_name)

    train_dataset = MyDataset(data=raw_train_dataset.get_data("train"))
    dev_dataset = MyDataset(data=raw_train_dataset.get_data("dev"))
    test_dataset = MyDataset(data=raw_test_dataset.get_data("test"))

    return train_dataset, dev_dataset, test_dataset


def prepare_data(args, model, datasets):
    train_dataset, dev_dataset, test_dataset = datasets

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        collate_fn=(lambda batch: my_collate(model, batch)),
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

    d_p = []
    d_s = []
    d_l = []
    t_p = []
    t_s = []
    t_l = []
    print("Running experiment {}".format(args.experiment_id))

    early_stop = EarlyStopping(
        monitor="dev/pearson", patience=4, verbose=True, mode="max"
    )

    trainer = pl.Trainer(
        devices=args.gpus,
        callbacks=[early_stop],
        # limit_train_batches=5,
        # limit_val_batches=2,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
        enable_checkpointing=False,
        max_epochs=args.max_train_epochs,
    )
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, train_dataloader, dev_dataloader)

    result_dev = trainer.test(model, dev_dataloader)
    result_test = trainer.test(model, test_dataloader)

    with open("results_{}.json".format(args.experiment_id), "w") as f:
        json.dump(result_test[0], f, indent=4, sort_keys=True)

    d_p.append(result_dev[0]["test/pearson"])
    d_s.append(result_dev[0]["test/spearman"])
    d_l.append(result_dev[0]["test/avg_loss"])
    t_p.append(result_test[0]["test/pearson"])
    t_s.append(result_test[0]["test/spearman"])
    t_l.append(result_test[0]["test/avg_loss"])

    print("Done, writing results...")
    result = {}
    result["dev_pearson"] = sum(d_p)
    result["dev_spearman"] = sum(d_s)
    result["dev_loss"] = sum(d_l)
    result["test_pearson"] = sum(t_p)
    result["test_spearman"] = sum(t_s)
    result["test_loss"] = sum(t_l)

    with open("results_of_{}.json".format(args.model_name.replace("/", "_")), "w") as f:
        json.dump(result, f, indent=4, sort_keys=True)

    return result


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
    test_dataset_name: str = "ro-sts"

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            self.__setattr__(key, value)


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


if __name__ == "__main__":
    GRID_SEARCH = {
        "max_train_epochs": [1],  # remove when running for real
        "train_dataset_name": [
            "ro-sts",
            "biblical_01",
        ],
        "model_name": [
            "dumitrescustefan/bert-base-romanian-uncased-v1",
            "readerbench/RoBERT-small",
            "readerbench/RoBERT-base",
        ],
        "loss_function": [],
    }

    # # auto - to run on a cluster
    # EXPERIEMNTS = get_experiments(GRID_SEARCH)

    # manual - to run on local
    EXPERIEMNTS = [
        {
            "max_train_epochs": 1,
        }
    ]

    for i, experiment in enumerate(EXPERIEMNTS):
        experiment["experiment_id"] = i
        run_experiment(experiment)
