"""Convert sampling results to COCO format.

This script converts generated images and captions into COCO dataset format
for evaluation and downstream tasks.
"""

import os
import json


def main(args):
    image_dir = os.path.join(args.sampling_root, "images")
    prompt_dir = os.path.join(args.sampling_root, "prompts")

    to_json = []
    for i in range(args.num_queries):
        image_path = os.path.join(image_dir, f"{i}.png")
        text_path = os.path.join(prompt_dir, f"{i}.txt")

        # read text
        with open(text_path, "r") as f:
            prompt = f.read().strip()
        
        if "cc3m" in args.sampling_root:
            to_json.append({"caption": prompt, "image_id": i, "image_name": str(i)+".png"})
        else:
            to_json.append({"caption": prompt, "image_id": i, "image": str(i)+".png"})
        
    if "flickr30k" in args.sampling_root:
        with open(os.path.join(args.sampling_root, "flickr30k_train.json"), "w") as f:
            json.dump(to_json, f)
            
    elif "coco" in args.sampling_root:
        with open(os.path.join(args.sampling_root, "coco_train.json"), "w") as f:
            json.dump(to_json, f)

    elif "cc3m" in args.sampling_root:
        with open(os.path.join(args.sampling_root, "captions_w_image_id.json"), "w") as f:
            json.dump(to_json, f)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling_root", type=str, default="stablediff/sampling_results/coco/")
    parser.add_argument("--num_queries", type=int, default=500)
    # parser.add_argument("--save_dir", type=str, default="cifar10")
    # parser.add_argument("--image_size", type=int, default=256)
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