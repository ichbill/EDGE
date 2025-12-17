"""Convert sampling results to Flickr30k format.

This script converts generated images and captions into Flickr30k dataset format
for evaluation and downstream tasks.
"""

import os
import json


def main(args):
    image_dir = os.path.join(args.sampling_root, "images")
    prompt_dir = os.path.join(args.sampling_root, "prompts")

    txt_files = sorted([f for f in os.listdir(prompt_dir) if f.endswith(".txt")], key=lambda x: int(x.split('.')[0]))

    to_json = []
    for i, prompt_path in enumerate(txt_files):
        height = width = args.image_size
        ratio = height / width
        path = os.path.join(image_dir, f"{i}.png")
        assert prompt_path == f"{i}.txt"
        with open(os.path.join(prompt_dir, prompt_path), "r") as f:
            prompt = f.read().strip()

        to_json.append({"height": height, "width": width, "ratio": ratio, "image": path, "caption": prompt, "image_id": i})
    
    with open(os.path.join(args.sampling_root, "captions.json"), "w") as f:
        json.dump(to_json, f)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling_root", type=str, default="label-prompt/CIFAR-10_labels.txt")
    # parser.add_argument("--save_dir", type=str, default="cifar10")
    parser.add_argument("--image_size", type=int, default=256)
    # parser.add_argument("--prompt_engineering", action="store_true")
    args = parser.parse_args()

    # if args.dataset == "cifar10":
    #     args.label_file_path = "label-prompt/CIFAR-10_labels.txt"
    # elif args.dataset == "cifar100":
    #     args.label_file_path = "label-promptcaption_pathcifar-100.txt"
    # elif args.dataset == "tinyimagenet":
    #     args.label_file_path = "label-prompt/tinyimgnt-label.txt"
    # elif args.dataset == "imagenet":
    #     args.label_file_path = "label-prompt/imagenet-classes.txt"
    # elif args.dataset == "flickr30k":
    #     args.label_file_path = "../data/flickr30k/captions/converted_flickr30k_train.json"
    # else:
    #     raise ValueError("Dataset not supported.")

    time_start = time.time()
    main(args)
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")