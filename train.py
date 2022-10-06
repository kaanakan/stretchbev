import os
import socket
import time

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

from stretchbev.config import get_parser, get_cfg
from stretchbev.data import prepare_dataloaders
from stretchbev.trainer import TrainingModule


def main():
    args = get_parser().parse_args()
    cfg = get_cfg(args)

    trainloader, valloader = prepare_dataloaders(cfg)
    model = TrainingModule(cfg.convert_to_dict())
    model.len_loader = len(trainloader)

    if cfg.PRETRAINED.LOAD_WEIGHTS:
        # Load single-image instance segmentation model.
        pretrained_model_weights = torch.load(
            os.path.join('.', cfg.PRETRAINED.PATH), map_location='cpu'
        )['state_dict']

        new_dict = {key: val for (key, val) in pretrained_model_weights.items() if 'decoder' not in key}
        model.load_state_dict(new_dict, strict=False)
        print(f'Loaded single-image model weights from {cfg.PRETRAINED.PATH}')

    save_dir = os.path.join(
        cfg.LOG_DIR, time.strftime('%d%B%Yat%H:%M:%S%Z') + '_' + socket.gethostname() + '_' + cfg.TAG
    )
    checkpoint_callback = ModelCheckpoint(dirpath='weights', filename='stretchbev-{epoch:02d}', save_top_k=-1)
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=save_dir)
    trainer = pl.Trainer(
        gpus=cfg.GPUS,
        accelerator='ddp',
        precision=cfg.PRECISION,
        sync_batchnorm=True,
        gradient_clip_val=cfg.GRAD_NORM_CLIP,
        max_epochs=cfg.EPOCHS,
        weights_summary='full',
        logger=tb_logger,
        log_every_n_steps=cfg.LOGGING_INTERVAL,
        plugins=DDPPlugin(find_unused_parameters=True),
        profiler='simple',
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, trainloader, valloader)


if __name__ == "__main__":
    main()
