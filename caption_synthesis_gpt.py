"""Caption Synthesis using GPT-4 Vision for EDGE-VLDD.

Usage:
    export OPENAI_API_KEY="your-api-key-here"
    python caption_synthesis_gpt.py --dataset coco \
        --label_file_path ./data/coco/annotations.json \
        --cpi 2 \
        --save_tag experiment_name
"""

import os
import time
import json
import base64
from tqdm import tqdm
from PIL import Image
from openai import OpenAI

# Initialize OpenAI client with API key from environment
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def gen_label_list(args):
    with open(args.label_file_path, "r") as f:
        captions = json.load(f)

    images = []
    labels = []

    if args.dataset == "flickr30k":
        raise NotImplementedError("Flickr30k not supported yet.")
        labels = [caption["prompt"] for caption in captions]
    
    elif args.dataset == "coco":
        for caption in captions:
            labels.append(caption["caption"])
            images.append(caption["image"])

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
        
        # Prompt for generating diverse captions
        prompt = "Describe the image briefly in a short sentence. Do not start with 'the image'"

        base64_image = encode_image(image_path)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": prompt,
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url":  f"data:image/jpeg;base64,{base64_image}"
                    },
                    },
                ],
                }
            ],
            n=args.cpi-1
        )
        
        captions = []
        captions.append(labels[i])
        for j in range(args.cpi-1):
            caption = response.choices[j].message.content
            captions.append(caption)

        for c in range(args.cpi):
            image = Image.open(image_path)
            image.save(os.path.join(img_dir, f"{i*args.cpi+c}.png"))
            with open(os.path.join(prompt_dir, f"{i*args.cpi+c}.txt"), "w") as f:
                f.write(captions[c])
            if args.dataset == "coco":
                annotations.append({"image_id": i*args.cpi+c, "image": f"{i*args.cpi+c}.png", "caption": captions[c]})
            elif args.dataset == "cc3m":
                annotations.append({"image_id": i*args.cpi+c, "image_name": f"{i*args.cpi+c}.png", "caption": captions[c]})

    if args.dataset == "coco":
        save_filename = "coco_train.json"
    elif args.dataset == "cc3m":
        save_filename = "captions_w_image_id.json"

    with open(os.path.join(save_dir, save_filename), "w") as f:
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

    print(f"Generating {args.cpi} captions per image ({args.cpi-1} synthetic + 1 original)")

    time_start = time.time()
    main(args)
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")