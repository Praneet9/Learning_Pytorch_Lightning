import torch
import torch.nn as nn
from torch.optim import Adam
from model import Model
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1


class Classifier(pl.LightningModule):
    def __init__(self, classes, lr):
        super().__init__()
        self.lr = lr
        self.model = Model()
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(multiclass=True, num_classes=classes)
        self.f1_score = F1(multiclass=True, num_classes=classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._step(batch)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        self.log_dict(
            {"Training Loss": loss, "Accuracy": accuracy, "F1 Score": f1_score},
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._step(batch)
        self.log("Validation Loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._step(batch)
        self.log("Test Loss", loss)
        return loss

    def _step(self, batch):
        x = batch["image"]
        y = batch["label"]
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def predict_step(self, batch, batch_idx):
        x = batch["image"]
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
