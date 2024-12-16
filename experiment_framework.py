import json
import os
import random
from pprint import pprint

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from googletrans import Translator
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from scipy.stats import spearmanr
from scipy.stats.stats import pearsonr
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer

from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
translator = Translator()


class AnglELoss(nn.Module):
    """
    Configure AngleLoss.

    :param cosine_w: float. weight for cosine_loss. Default 1.0
    :param ibn_w: float. weight for contrastive loss. Default 1.0
    :param angle_w: float. weight for angle loss. Default 1.0
    :param cosine_tau: float. tau for cosine loss. Default 20.0
    :param ibn_tau: float. tau for contrastive loss. Default 20.0
    :param angle_tau: float. tau for angle loss. Default 20.0
    :param angle_pooling_strategy: str. pooling strategy for angle loss. Default'sum'.
    :param dataset_format: Optional[str]. Default None.
    """

    def __init__(
        self,
        cosine_w: float = 0.0,
        ibn_w: float = 20.0,
        angle_w: float = 1.0,
        cosine_tau: float = 20.0,
        ibn_tau: float = 20.0,
        angle_tau: float = 20.0,
        angle_pooling_strategy: str = "sum",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cosine_w = cosine_w
        self.ibn_w = ibn_w
        self.angle_w = angle_w
        self.cosine_tau = cosine_tau
        self.ibn_tau = ibn_tau
        self.angle_tau = angle_tau
        self.angle_pooling_strategy = angle_pooling_strategy

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.mse = nn.MSELoss()

    def categorical_crossentropy(
        self, y_true: torch.Tensor, y_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute categorical crossentropy

        :param y_true: torch.Tensor, ground truth
        :param y_pred: torch.Tensor, model output
        :param from_logits: bool, `True` means y_pred has not transformed by softmax, default True

        :return: torch.Tensor, loss value
        """
        return -(F.log_softmax(y_pred, dim=1) * y_true).sum(dim=1)

    def cosine_loss(
        self, y_true: torch.Tensor, y_pred: torch.Tensor, tau: float = 20.0
    ) -> torch.Tensor:
        """
        Compute cosine loss

        :param y_true: torch.Tensor, ground truth.
            The y_true must be zigzag style, such as [x[0][0], x[0][1], x[1][0], x[1][1], ...], where (x[0][0], x[0][1]) stands for a pair.
        :param y_pred: torch.Tensor, model output.
            The y_pred must be zigzag style, such as [o[0][0], o[0][1], o[1][0], o[1][1], ...], where (o[0][0], o[0][1]) stands for a pair.
        :param tau: float, scale factor, default 20

        :return: torch.Tensor, loss value
        """  # NOQA
        # modified from: https://github.com/bojone/CoSENT/blob/124c368efc8a4b179469be99cb6e62e1f2949d39/cosent.py#L79
        y_true = y_true[::2, 0]
        y_true = (y_true[:, None] < y_true[None, :]).float()
        y_pred = F.normalize(y_pred, p=2, dim=1)
        y_pred = torch.sum(y_pred[::2] * y_pred[1::2], dim=1) * tau
        y_pred = y_pred[:, None] - y_pred[None, :]
        y_pred = (y_pred - (1 - y_true) * 1e12).view(-1)
        zero = torch.Tensor([0]).to(y_pred.device)
        y_pred = torch.concat((zero, y_pred), dim=0)
        return torch.logsumexp(y_pred, dim=0)

    def angle_loss(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        tau: float = 1.0,
        pooling_strategy: str = "sum",
    ):
        """
        Compute angle loss

        :param y_true: torch.Tensor, ground truth.
            The y_true must be zigzag style, such as [x[0][0], x[0][1], x[1][0], x[1][1], ...], where (x[0][0], x[0][1]) stands for a pair.
        :param y_pred: torch.Tensor, model output.
            The y_pred must be zigzag style, such as [o[0][0], o[0][1], o[1][0], o[1][1], ...], where (o[0][0], o[0][1]) stands for a pair.
        :param tau: float, scale factor, default 1.0

        :return: torch.Tensor, loss value
        """  # NOQA
        y_true = y_true[::2, 0]
        y_true = (y_true[:, None] < y_true[None, :]).float()

        y_pred_re, y_pred_im = torch.chunk(y_pred, 2, dim=1)
        a = y_pred_re[::2]
        b = y_pred_im[::2]
        c = y_pred_re[1::2]
        d = y_pred_im[1::2]

        # (a+bi) / (c+di)
        # = ((a+bi) * (c-di)) / ((c+di) * (c-di))
        # = ((ac + bd) + i(bc - ad)) / (c^2 + d^2)
        # = (ac + bd) / (c^2 + d^2) + i(bc - ad)/(c^2 + d^2)
        z = torch.sum(c**2 + d**2, dim=1, keepdim=True)
        re = (a * c + b * d) / z
        im = (b * c - a * d) / z

        dz = torch.sum(a**2 + b**2, dim=1, keepdim=True) ** 0.5
        dw = torch.sum(c**2 + d**2, dim=1, keepdim=True) ** 0.5
        re /= dz / dw
        im /= dz / dw

        y_pred = torch.concat((re, im), dim=1)
        if pooling_strategy == "sum":
            pooling = torch.sum(y_pred, dim=1)
        elif pooling_strategy == "mean":
            pooling = torch.mean(y_pred, dim=1)
        else:
            raise ValueError(f"Unsupported pooling strategy: {pooling_strategy}")
        y_pred = torch.abs(pooling) * tau  # absolute delta angle
        y_pred = y_pred[:, None] - y_pred[None, :]
        y_pred = (y_pred - (1 - y_true) * 1e12).view(-1)
        zero = torch.Tensor([0]).to(y_pred.device)
        y_pred = torch.concat((zero, y_pred), dim=0)
        return torch.logsumexp(y_pred, dim=0)

    def in_batch_negative_loss(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        tau: float = 20.0,
        negative_weights: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute in-batch negative loss, i.e., contrastive loss

        :param y_true: torch.Tensor, ground truth.
            The y_true must be zigzag style, such as [x[0][0], x[0][1], x[1][0], x[1][1], ...], where (x[0][0], x[0][1]) stands for a pair.
        :param y_pred: torch.Tensor, model output.
            The y_pred must be zigzag style, such as [o[0][0], o[0][1], o[1][0], o[1][1], ...], where (o[0][0], o[0][1]) stands for a pair.
        :param tau: float, scale factor, default 20.0
        :param negative_weights: float, negative weights, default 0.0

        :return: torch.Tensor, loss value
        """  # NOQA
        device = y_true.device

        def make_target_matrix(y_true: torch.Tensor):
            idxs = torch.arange(0, y_pred.shape[0]).int().to(device)
            y_true = y_true.int()
            idxs_1 = idxs[None, :]
            idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]

            idxs_1 *= y_true.T
            idxs_1 += (y_true.T == 0).int() * -2

            idxs_2 *= y_true
            idxs_2 += (y_true == 0).int() * -1

            y_true = (idxs_1 == idxs_2).float()
            return y_true

        neg_mask = make_target_matrix(y_true == 0)

        y_true = make_target_matrix(y_true)

        # compute similarity
        y_pred = F.normalize(y_pred, dim=1, p=2)
        similarities = y_pred @ y_pred.T  # dot product
        similarities = similarities - torch.eye(y_pred.shape[0]).to(device) * 1e12
        similarities = similarities * tau

        if negative_weights > 0:
            similarities += neg_mask * negative_weights

        return self.categorical_crossentropy(y_true, similarities).mean()

    def forward(
        self,
        sentence_1: torch.Tensor,
        sentence_2: torch.Tensor,
        cosines,
        sim: torch.Tensor,
    ):
        loss = 0.0
        if self.cosine_w > 0:
            loss += self.cosine_w * self.cosine_loss(
                sentence_1, sentence_2, self.cosine_tau
            )
        if self.ibn_w > 0:
            loss += self.ibn_w * self.in_batch_negative_loss(
                sentence_1, sentence_2, self.ibn_tau
            )
        if self.angle_w > 0:
            loss += self.angle_w * self.angle_loss(
                sentence_1,
                sentence_2,
                self.angle_tau,
                pooling_strategy=self.angle_pooling_strategy,
            )
        return loss


class CosineSimilarityMSELoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.mse = nn.MSELoss()

    def forward(
        self,
        sentence_1: torch.Tensor,
        sentence_2: torch.Tensor,
        cosines,
        sim: torch.Tensor,
    ):
        loss = self.mse(cosines, sim)
        return loss


class TransformerModel(pl.LightningModule):
    def __init__(
        self,
        model_name="dumitrescustefan/bert-base-romanian-cased-v1",
        lr=2e-05,
        model_max_length=512,
        loss_function="MSE",
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

        self.loss_fct = self.select_loss_function(loss_function)
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

    def select_loss_function(self, loss_function):
        if loss_function == "MSE":
            return CosineSimilarityMSELoss()
        if loss_function == "AnglE":
            return AnglELoss()

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
        loss = self.loss_fct(pooled_sentence1, pooled_sentence2, cosines, sim)
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
        collate_fn=(lambda batch: my_collate(model, batch, translate=args.data_augmentation_translate_data)),
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

    results_location = os.path.join(trainer.logger.log_dir, "results.json")

    with open(results_location, "w") as f:
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
    data_augmentation_translate_data: bool = False

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
        # "max_train_epochs": [20],  # remove when running for real
        # "train_dataset_name": [
        #     "ro-sts",
        #     # "biblical_01", # comment biblical_01 because it is huuuge :D
        #     # ["ro-sts", "biblical_01"],
        # ],
        "accumulate_grad_batches": [32],
        "model_name": [
            "dumitrescustefan/bert-base-romanian-cased-v1",
            "dumitrescustefan/bert-base-romanian-uncased-v1",
            "readerbench/RoBERT-small",
            "readerbench/RoBERT-base",
        ],
        "loss_function": ["MSE", "AnglE"],
        "data_augmentation_translate_data": [False, True],
    }

    # # auto - to run on a cluster
    EXPERIEMNTS = get_experiments(GRID_SEARCH)

    # manual - to run on local
    EXPERIEMNTS = [
        {
            "max_train_epochs": 1,
            "loss_function": "AnglE",
            "data_augmentation_translate_data": True,
        },
    ]

    for i, experiment in enumerate(EXPERIEMNTS):
        experiment["experiment_id"] = i
        run_experiment(experiment)
