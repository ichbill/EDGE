"""Image sampling script using Stable Diffusion for dataset generation.

Usage:
    Single GPU:
        python sampling.py --dataset coco --num_queries 500
    
    Multi-GPU:
        CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 accelerate launch --num_processes=6 sampling.py --dataset coco --num_queries 500
"""

import os
import time
import json
import random

import torch
from diffusers import StableDiffusionPipeline
from accelerate import Accelerator

def gen_label_list(args):
    with open(args.label_file_path, "r") as f:
        captions = json.load(f)

    captions = random.sample(captions, args.num_queries)

    if args.dataset == "flickr30k":
        labels = [caption["prompt"] for caption in captions]
    elif args.dataset == "coco":
        labels = [caption["caption"] for caption in captions]
    elif args.dataset == "cc3m":
        labels = [caption["caption"] for caption in captions]
    else:
        labels = [caption.get("caption", caption.get("prompt", "")) for caption in captions]
    
    return labels


def main(args):
    accelerator = Accelerator()

    labels = gen_label_list(args)
    if accelerator.is_main_process:
        print(f"Sampling {len(labels)} prompts with {args.ipc} images per prompt...")

    # Load Stable Diffusion model
    pipe = StableDiffusionPipeline.from_pretrained(
        os.path.join("checkpoints", args.sd_model),
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    # Disable safety checker for research purposes
    pipe.safety_checker = lambda images, clip_input: (images, [False] * len(images))
    pipe.to(accelerator.device)

    # Create output directories
    save_dir = os.path.join("sampling_results", args.dataset, args.save_tag)
    img_dir = os.path.join(save_dir, "images")
    prompt_dir = os.path.join(save_dir, "prompts")
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(prompt_dir, exist_ok=True)
    
    if accelerator.is_main_process:
        print(f"Saving to {save_dir}")
    
    acc_round = 0
    with accelerator.split_between_processes(labels) as prompts:
        for prompt in prompts:
            # Apply prompt engineering if requested
            fine_prompt = f"A picture of a {prompt}" if args.prompt_engineering else prompt
            
            # Generate images
            images = pipe(fine_prompt, num_images_per_prompt=args.ipc).images
            
            # Save images and prompts
            for i, image in enumerate(images):
                image_index = (
                    acc_round * accelerator.num_processes * args.ipc + 
                    accelerator.process_index * args.ipc + i
                )
                image.save(os.path.join(img_dir, f"{image_index}.png"))
                
                with open(os.path.join(prompt_dir, f"{image_index}.txt"), "w") as f:
                    f.write(prompt)
            
            acc_round += 1

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--label_file_path", type=str, default="label-prompt/CIFAR-10_labels.txt")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--ipc", type=int, default=1)
    parser.add_argument("--num_queries", type=int, default=100)
    parser.add_argument("--prompt_engineering", action="store_true")
    parser.add_argument("--sd_model", type=str, default="mse_03")
    parser.add_argument("--save_tag", type=str, default="test")
    args = parser.parse_args()

    # Set dataset-specific label file paths
    dataset_paths = {
        "cifar10": "label-prompt/CIFAR-10_labels.txt",
        "cifar100": "label-prompt/cifar-100.txt",
        "tinyimagenet": "label-prompt/tinyimgnt-label.txt",
        "imagenet": "label-prompt/imagenet-classes.txt",
        "flickr30k": "data/flickr30k/captions/converted_flickr30k_train.json",
        "coco": "data/coco/coco_karpathy_train.json",
        "cc3m": "data/CC3M/captions_w_image_id.json",
    }
    
    if args.dataset in dataset_paths:
        args.label_file_path = dataset_paths[args.dataset]
    
    time_start = time.time()
    main(args)
    time_end = time.time()
    
    print(f"\nCompleted in {time_end - time_start:.2f} seconds")