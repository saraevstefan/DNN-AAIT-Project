import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch.utils.data.dataset import Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoTokenizer,
)


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
        if model_name == "BlackKakapo/t5-base-paraphrase-ro-v2":
            self.model = T5ForConditionalGeneration.from_pretrained(model_name).encoder
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        else:
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


if __name__ == "__main__":
    model = TransformerModel("BlackKakapo/t5-base-paraphrase-ro-v2")

    s1 = "Ana are niste mere"
    inp = model.tokenizer(s1, return_tensors="pt")
    a = 1
