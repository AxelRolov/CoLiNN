import logging
import os.path as osp
from typing import Dict, Any

import click
import torch
import tomllib  # For Python 3.11+; for older versions, use `import toml` and `pip install toml`
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging, ModelSummary

from colinn.modules import CoLiNN
from colinn.preprocessing import CoLiNNData, process_bbs


def main_pipeline(config: Dict[str, Any], mode: str) -> None:
    """
    Main function to execute the training or prediction pipeline.

    :param config: Configuration dictionary loaded from the TOML file.
    :param mode: Operation mode (either 'training' or 'prediction').
    """
    logging.basicConfig(
        filename=config["Train"]["log_file"],
        filemode="w",
        level=config["Train"]["log_level"],
        format='%(asctime)s:%(levelname)s:%(message)s'
    )

    # Process building blocks
    name: str = config["Data"]["name"]
    bb_file: str = osp.join(f"{config['Data'][mode]['root_path']}/raw/", f"{name}_bbs.svm")
    processed_bb_path: str = config["Model"]["bbs_pyg_path"]

    # Preprocess building blocks if necessary
    if not osp.exists(processed_bb_path) and bb_file:
        logging.debug("Started preprocessing building blocks")
        process_bbs(bb_file, name, f"{config['Data'][mode]['root_path']}/processed/")

    # Prepare data for the model
    data: CoLiNNData = CoLiNNData(
        root=config["Data"][mode]["root_path"],
        batch_size=config['Model']['batch_size'],
        name=config["Data"]["name"],
        mode=mode
    )

    # Set paths if not available in the config
    config['Model']['bbs_pyg_path'] = config['Model'].get('bbs_pyg_path', None)
    config['Model']['bbs_embed_path'] = config['Model'].get('bbs_embed_path', None)

    # Set random seed for reproducibility
    seed_everything(42, workers=True)

    # Initialize the model
    model: CoLiNN = CoLiNN(
        batch_size=config['Model']['batch_size'],
        vector_dim=config['Model']['vector_dim'],
        num_reactions=config['Model']['num_reactions'],
        num_conv_layers=config['Model']['num_conv_layers'],
        num_gtm_nodes=config['Model']['num_gtm_nodes'],
        bbs_pyg_path=config['Model']['bbs_pyg_path'],
        bbs_embed_path=config['Model']['bbs_embed_path'],
        lr=config['Train']['lr']
    )

    # Move model to the appropriate device (GPU if available, otherwise CPU)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize training components
    lr_monitor: LearningRateMonitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint: ModelCheckpoint = ModelCheckpoint(
        dirpath=config["Train"]["weights_path"],
        filename=config["Train"]["weights_name"],
        monitor="val_loss",
        mode="min"
    )
    swa: StochasticWeightAveraging = StochasticWeightAveraging(
        swa_lrs=config["Train"]["lr"],
        swa_epoch_start=0.95
    )

    # Set up trainer
    trainer: Trainer = Trainer(
        accelerator='gpu',
        devices=[0],
        max_epochs=config["Train"]["max_epochs"],
        callbacks=[lr_monitor, checkpoint, swa, ModelSummary(2)],
        precision=16,
        gradient_clip_val=1.0,
        log_every_n_steps=50
    )

    # Perform training or prediction based on mode
    if mode == "training":
        trainer.fit(model, data)
    else:
        # Load model from checkpoint for prediction
        model = CoLiNN.load_from_checkpoint(
            checkpoint_path=f"{config['Train']['weights_path']}/{config['Train']['weights_name']}.ckpt",
            batch_size=config['Model']['batch_size'],
            vector_dim=config['Model']['vector_dim'],
            num_reactions=config['Model']['num_reactions'],
            num_conv_layers=config['Model']['num_conv_layers'],
            num_gtm_nodes=config['Model']['num_gtm_nodes'],
            bbs_pyg_path=config['Model']['bbs_pyg_path'],
            bbs_embed_path=config['Model']['bbs_embed_path'],
            lr=config['Train']['lr']
        )
        model.to(device)

        # Generate predictions
        predictions = trainer.predict(model, datamodule=data)
        all_predictions: torch.Tensor = torch.cat(predictions, dim=0)
        torch.save(all_predictions, osp.join(config["Data"][mode]["root_path"], 'processed/predictions.pt'))


@click.command()
@click.option("--config", "config_path", required=True, help="Path to the config TOML file.",
              type=click.Path(exists=True))
@click.option("--mode", default="training", type=click.Choice(["training", "prediction"]),
              help="Mode of operation (training or prediction).")
def main(config_path: str, mode: str) -> None:
    """
    Main function that loads the configuration and runs the main pipeline.

    :param config_path: Path to the TOML config file.
    :param mode: Mode of operation (training or prediction).
    """
    config_path = osp.abspath(config_path)

    if osp.exists(config_path) and osp.isfile(config_path):
        # Load the TOML config file
        with open(config_path, 'rb') as config_file:
            config: Dict[str, Any] = tomllib.load(config_file)

        if config is not None:
            main_pipeline(config, mode)
    else:
        raise ValueError("The config file does not exist.")


if __name__ == '__main__':
    main()
