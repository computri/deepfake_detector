import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything

from data_utils import CIFAKEDataModule
from model_pipelines import DeepFakeClassifier
from model_utils import build_discriminator, image_size
import torch
from pytorch_lightning.callbacks import ModelCheckpoint



@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg: DictConfig):
    seed_everything(cfg.seed, workers=True)

    torch.set_float32_matmul_precision('medium')

    discriminator = build_discriminator(
        backbone=cfg.discriminator.backbone,
        pretrained=cfg.discriminator.pretrained,
        fine_tune=cfg.discriminator.fine_tune
    )
    
    datamodule = CIFAKEDataModule(
        data_dir=cfg.data.root_dir, 
        batch_size=cfg.data.batch_size, 
        num_workers=cfg.data.num_workers, 
        image_size=image_size(cfg.discriminator.backbone)
    )

    model = DeepFakeClassifier(
        model=discriminator, 
        learning_rate=cfg.train.learning_rate
    )

    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",           
            save_top_k=1,                 
            mode="min",                   
            dirpath=f"checkpoints/{cfg.discriminator.backbone}" if cfg.discriminator.checkpoint_save_path is None else cfg.discriminator.checkpoint_save_path,        
            filename="best-epoch" 
        )
    ]

    trainer = Trainer(
        logger=None, 
        num_sanity_val_steps=0,
        accelerator=cfg.device,
        devices=1 if cfg.device == "gpu" else None,
        max_epochs=cfg.train.epochs,
        callbacks=callbacks,   
    )

    if cfg.discriminator.compile:
        print("Compiling the model for performance optimization.")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        model = torch.compile(model)
        
    trainer.fit(model, datamodule=datamodule)

    
    trainer.validate(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

if __name__ == "__main__":
    main()