"""Caption Synthesis using LLaVA for EDGE-VLDD.

This module uses LLaVA (Large Language and Vision Assistant) to generate 
diverse paraphrases of image captions for data augmentation.

Usage:
    CUDA_VISIBLE_DEVICES=0 python caption_synthesis_llava.py \
        --dataset coco \
        --label_file_path ./data/coco/annotations.json \
        --cpi 2 \
        --save_tag experiment_name
"""

import os
import subprocess
import time
import json
from tqdm import tqdm
from PIL import Image

# Model configuration
MODEL_PATH = "liuhaotian/llava-v1.5-7b"

def gen_label_list(args):
    with open(args.label_file_path, "r") as f:
        captions = json.load(f)

    images = []
    labels = []

    if args.dataset == "flickr30k":
        raise NotImplementedError("Flickr30k not supported yet.")
        labels = [caption["prompt"] for caption in captions]
    
    elif args.dataset == "coco" :
        for caption in captions:
            labels.append(caption["caption"])
            images.append(caption["image"])

    elif args.dataset == "cc3m":
        for caption in captions:
            labels.append(caption["caption"])
            images.append(caption["image_name"])

    return images, labels

def main(args):
    images, labels = gen_label_list(args)

    save_dir = os.path.join("sampling_results", args.dataset, args.save_tag)
    print(f"Saving to {save_dir}")
    img_dir = os.path.join(save_dir, "images")
    prompt_dir = os.path.join(save_dir, "prompts")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(prompt_dir, exist_ok=True)
    
    annotations = []
    root_path = os.path.dirname(args.label_file_path)
    image_folder = os.path.join(root_path, "images")
    
    for i in tqdm(range(len(images))):
        image_file = images[i]
        image_path = os.path.join(image_folder, image_file)
        assert i == int(image_file.split(".")[0])

        # Build command for LLaVA subprocess
        command = [
            "python", "-m", "llava.serve.cli",
            "--model-path", MODEL_PATH,
            "--image-file", image_path
        ]
        
        if args.load_4bit:
            command.append("--load-4bit")

        # Call LLaVA model
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Generate prompt based on requested captions per image
        if args.cpi > 2:
            prompt = f"Describe the image briefly in {args.cpi-1} different ways."
        else:
            prompt = "Describe the image in one sentence."
        
        process.stdin.write(f"{prompt}\n".encode())
        process.stdin.flush()

        # Get response from model
        output, error = process.communicate()
        caption = output.decode().strip()
        
        # Parse captions from output
        if args.cpi > 2:
            captions = [". ".join(c.split(". ")[1:]) for c in caption.split("\n") if c][:-1]
        else:
            captions = [c for c in caption.split("\n") if c]
            captions.pop(-1)
            captions = [captions[0].split("ASSISTANT: ")[-1]]
        
        # Append original caption
        captions.append(labels[i])


        for c in range(args.cpi):
            image = Image.open(image_path)
            image.save(os.path.join(img_dir, f"{i*args.cpi+c}.png"))
            with open(os.path.join(prompt_dir, f"{i*args.cpi+c}.txt"), "w") as f:
                f.write(captions[c])
            annotations.append({"image_id": i*args.cpi+c, "image": f"{i*args.cpi+c}.png", "caption": captions[c]})
        # caption = caption.split("\n")[0]
        # caption = caption.split("USER: ASSISTANT: ")[-1]
        
        # captions.append(caption)

        # Save the image and prompt
        # image = Image.open(image_path)
        # image.save(os.path.join(img_dir, f"{i*args.cpi}.png"))
        # # save the prompt
        # with open(os.path.join(prompt_dir, f"{i*args.cpi}.txt"), "w") as f:
        #     f.write(labels[i])
        # annotations.append({"image_id": i*args.cpi, "image": f"{i*args.cpi}.png", "caption": labels[i]})
        
        # image.save(os.path.join(img_dir, f"{i*args.cpi+1}.png"))
        # with open(os.path.join(prompt_dir, f"{i*args.cpi+1}.txt"), "w") as f:
        #     f.write(caption)
        # annotations.append({"image_id": i*args.cpi+1, "image": f"{i*args.cpi+1}.png", "caption": caption})

    if args.dataset == "coco":
        with open(os.path.join(save_dir, "coco_train.json"), "w") as f:
            json.dump(annotations, f)
    elif args.dataset == "cc3m":
        with open(os.path.join(save_dir, "captions_w_image_id.json"), "w") as f:
            json.dump(annotations, f)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--load_4bit", action="store_true")
    parser.add_argument("--label_file_path", type=str, default="./images")
    parser.add_argument("--cpi", type=int, default=2)
    parser.add_argument("--save_tag", type=str, default="test")
    args = parser.parse_args()

    time_start = time.time()
    main(args)
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")