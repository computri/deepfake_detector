import torch
import hydra
from omegaconf import DictConfig
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DiffusionPipeline,
)
from pathlib import Path
from PIL import Image
from tqdm import tqdm



@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    device = "cuda" if cfg.device == "gpu" else "cpu"
    classes = get_class_labels(["coco"])
    generate_images(classes, device=device, cfg=cfg.generator)


def get_class_labels(datasets):
    labels = []
    for dataset in datasets:
        with open(f'../data/{dataset}/classes.txt') as f:
            classes = f.read().splitlines()
            labels += classes
    
    return list(set(labels))

def get_generator(model_name: str, torch_dtype=torch.float16):
    """
    Returns a diffusion pipeline and its recommended height/width for generation.
    """
    # Default resolution
    default_res = (512, 512)

    # Choose pipeline + override recommended resolution
    if "xl" in model_name.lower():
        pipeline_class = StableDiffusionXLPipeline
        default_res = (1024, 1024)
    elif "stable-diffusion" in model_name.lower() or "runwayml" in model_name.lower():
        pipeline_class = StableDiffusionPipeline
        default_res = (512, 512)
    elif "deepfloyd" in model_name.lower() or "if" in model_name.lower():
        pipeline_class = DiffusionPipeline
        default_res = (1024, 1024)
    else:
        pipeline_class = DiffusionPipeline
        default_res = (512, 512)

    # Load pipeline
    pipeline = pipeline_class.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        use_safetensors=True
    )

    return pipeline, default_res

def generate_images(classes, device="cuda", cfg=None):
    # Config

    prompt = cfg.prompt.split("_BLANK_")
        
    # prompts = [f"A documentary photography of a {cl}" for cl in classes]
    prompts = [f"{prompt[0]}{cl}{prompt[1]}" for cl in classes]
    output_dir = Path("../data/generated_images")
    output_dir.mkdir(exist_ok=True)
    
    if cfg is None:
        num_images_per_prompt = 10
        guidance_scale = 7.5
        num_inference_steps = 30
    else:
        num_images_per_prompt = cfg.num_images_per_prompt
        guidance_scale = cfg.guidance_scale
        num_inference_steps = cfg.num_inference_steps

    # # Load SDXL pipeline
    # pipeline = StableDiffusionXLPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-xl-base-1.0",
    #     torch_dtype=torch.float16,
    #     use_safetensors=True,
    # )

    pipeline, resolution = get_generator(cfg.model_name, torch_dtype=torch.float16)
    pipeline.to(device)

    # Generate images
    for class_idx, prompt in enumerate(tqdm(prompts, desc="Generating images")):
        for i in range(num_images_per_prompt):
            image = pipeline(
                prompt=prompt,
                height=resolution[0],
                width=resolution[1],
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            ).images[0]

            
            # Save image
            filename = f"{classes[class_idx]}_{i+1}.jpg"
            
            image.save(output_dir / filename)


if __name__ == "__main__":
    main()

