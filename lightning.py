
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
        self.train_loader, self.val_loader, self.test_loader = load_datasets(
            int(config["training"]["batch_size"]),
            int(config["training"]["num_workers"]),
        )

        self.example_input_array = torch.randn(1, 3, 512, 512)

        self.example_images = [(
            self.val_loader.dataset[i][0].unsqueeze(0),
            self.val_loader.dataset[i][1].item()
        ) for i in range(5)]

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

        if self.current_epoch == 0:
            for i in range(len(self.example_images)):
                target = self.example_images[i][1]
                self.logger.experiment.add_image(
                    "Image {}, {} Years Old".format(i, target), self.example_images[i][0].squeeze(0)
                    , self.current_epoch, dataformats="CHW"
                )

        for i in range(len(self.example_images)):
            prediction = self.model(self.example_images[i][0].to(self.device)).item()
            target = self.example_images[i][1]
            self.logger.experiment.add_scalar(
                "Image {}, Target {} vs Prediction".format(i, int(target)),
                prediction, self.current_epoch
            )

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
            'lr_scheduler': self.scheduler,
            'monitor': 'val_loss'
        }
