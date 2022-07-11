
import pytorch_lightning as pl
import torch

from data_loading import load_datasets
from utils import load_model, load_criterion, load_optimizer, load_scheduler


class LitModel(pl.LightningModule):

    def __init__(self, config):
        super(LitModel, self).__init__()
        self.config = config
        self.model = load_model(config)
        self.optimizer = load_optimizer(self.model, config)
        self.scheduler = load_scheduler(self.optimizer, config)
        self.criterion = load_criterion(config)
        self.train_loader, self.val_loader, self.test_loader = load_datasets(int(config["training"]["batch_size"]))
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def training_step(self, batch, batch_idx):
        x, target = batch
        output = self.model(x)
        loss = self.criterion(output.view(-1), target.view(-1))
        self.log('train_loss', loss, on_step=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, target = batch
        output = self.model(x)
        loss = self.criterion(output.view(-1), target.view(-1))
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss, prog_bar=True)
        return {'val_loss': avg_loss}

    def test_step(self, batch, batch_idx):
        x, target = batch
        output = self.model(x)
        loss = self.criterion(output.view(-1), target.view(-1))
        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.log('test_loss', avg_loss, prog_bar=True)
        return {'test_loss': avg_loss}

    def configure_optimizers(self):
        return {
            'optimizer': self.optimizer,
            'scheduler': self.scheduler
        }


