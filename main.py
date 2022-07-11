import pytorch_lightning as pl
import torch.cuda
from pytorch_lightning.loggers import TensorBoardLogger

from lightning import LitModel
from utils import load_config


def main():
    """
    Main function
    """

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    config = load_config()

    lit_model = LitModel(config)

    logger = TensorBoardLogger("logs", name=config["experiment_name"])
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=-1,
        logger=logger,
        max_epochs=int(config["training"]["max_epochs"])
    )
    trainer.fit(model=lit_model)


if __name__ == "__main__":
    main()
