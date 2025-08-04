import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything

from data_utils import get_datamodule 
from model_pipelines import DeepFakeClassifier
from model_utils import build_discriminator, image_size
import torch

@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg: DictConfig):
    seed_everything(cfg.seed, workers=True)

    discriminator = build_discriminator(
        backbone=cfg.discriminator.backbone,
        pretrained=cfg.discriminator.pretrained,
        fine_tune=cfg.discriminator.fine_tune
    )
    
    datamodule = get_datamodule(cfg.data, image_size(cfg.discriminator.backbone))

    # Load weights
    model = DeepFakeClassifier.load_from_checkpoint(
        f"checkpoints/{cfg.discriminator.backbone}/best-epoch.ckpt" if cfg.discriminator.checkpoint_load_path is None else cfg.discriminator.checkpoint_load_path,        
        model=discriminator, 
        learning_rate=cfg.train.learning_rate,
        report_per_class_accuracy=False
    )

    trainer = Trainer(
        logger=None, 
        num_sanity_val_steps=0,
        accelerator=cfg.device,
        devices=1 if cfg.device == "gpu" else None,
        max_epochs=cfg.train.epochs,
    )

    if cfg.discriminator.compile:
        print("Compiling the model for performance optimization.")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        model = torch.compile(model)
        
    trainer.validate(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

if __name__ == "__main__":
    main()