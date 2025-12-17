"""Caption Synthesis using Llama for EDGE-VLDD.

This module uses Llama language model to generate diverse paraphrases of image captions,
enabling data augmentation for vision-language dataset distillation.

Usage:
    CUDA_VISIBLE_DEVICES=0 python caption_synthesis_llama.py --dataset coco \
        --image_root ./data/coco \
        --ann_root ./data/coco \
        --ckpt_dir /path/to/llama/checkpoint \
        --output_file ./synthesized_captions.json
"""

import os
import sys
import time
import json

from llama import Llama

from dataset import create_dataset

def main(args):
    """Generate diverse caption paraphrases using Llama.
    
    Args:
        args: Argument namespace containing configuration parameters
    """
    print("Loading dataset...")
    train_dataset, val_dataset, test_dataset = create_dataset(args)
    image_ids, captions = train_dataset.get_all_pairs()
    print(f"Loaded {len(captions)} captions from {args.dataset}")

    # Initialize Llama model
    print("Loading Llama model...")
    generator = Llama.build(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
        seed=args.seed,
    )
    print("Model loaded successfully")

    # Generate prompts for caption paraphrasing
    print(f"Generating {args.cpi} variants per caption...")
    prompts = []
    caption_indices = []  # Track which original caption each prompt corresponds to
    
    for idx, caption in enumerate(captions):
        clean_caption = caption.replace('\n', ' ').strip()
        for i in range(args.cpi):
            prompt = f"""Rewrite this image caption with different wording while keeping the same meaning:

{clean_caption} =>"""
            prompts.append(prompt)
            caption_indices.append(idx)
    
    print(f"Generated {len(prompts)} total prompts")

    # Generate paraphrases in batches
    print("Generating caption variants...")
    all_results = []
    batch_size = args.max_batch_size
    
    for i in range(0, len(prompts), batch_size):
        sub_batch = prompts[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size}")
        
        results = generator.text_completion(
            sub_batch,
            max_gen_len=args.max_gen_len,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        all_results.extend(results)
    
    # Organize results by original caption
    synthesized_data = []
    for idx in range(len(captions)):
        caption_variants = []
        for prompt_idx, cap_idx in enumerate(caption_indices):
            if cap_idx == idx:
                generated_text = all_results[prompt_idx]['generation']
                caption_variants.append(generated_text.strip())
        
        synthesized_data.append({
            'image_id': image_ids[idx],
            'original_caption': captions[idx],
            'synthesized_captions': caption_variants
        })
    
    # Save results
    output_path = args.output_file
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(synthesized_data, f, indent=2)
    
    print(f"\nSynthesized captions saved to {output_path}")
    print(f"Total original captions: {len(captions)}")
    print(f"Total synthesized captions: {len(prompts)}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate diverse caption paraphrases using Llama for vision-language dataset distillation'
    )
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='flickr30k', 
                        choices=['flickr30k', 'coco'], help='Dataset name')
    parser.add_argument('--image_root', type=str, default='distill_utils/data/Flickr30k/', 
                        help='Path to image root directory')
    parser.add_argument('--ann_root', type=str, default='./data/Flickr30k_ann/', 
                        help='Path to annotation root directory')
    parser.add_argument('--distill_image', type=str, default='../', 
                        help='Path to distilled images (for dataset loading compatibility)')
    parser.add_argument('--distill_ann', type=str, default='../sampling_results/flickr30k/', 
                        help='Path to distilled annotations (for dataset loading compatibility)')
    parser.add_argument('--image_size', type=int, default=224, 
                        help='Image size for dataset processing')
    parser.add_argument('--no_aug', action="store_true", default=False, 
                        help='Disable data augmentation during dataset loading')

    # Generation parameters
    parser.add_argument("--cpi", type=int, default=5, 
                        help='Captions per image: number of paraphrase variants to generate')
    parser.add_argument('--output_file', type=str, default='./synthesized_captions.json',
                        help='Path to save synthesized captions')

    # Llama model parameters
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help='Path to Llama model checkpoint directory')
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help='Path to Llama tokenizer model file')
    parser.add_argument("--max_seq_len", type=int, default=512, 
                        help='Maximum sequence length for the model')
    parser.add_argument("--max_gen_len", type=int, default=64, 
                        help='Maximum generation length for output captions')
    parser.add_argument("--max_batch_size", type=int, default=8, 
                        help='Maximum batch size for inference')
    parser.add_argument("--seed", type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument("--temperature", type=float, default=0.8, 
                        help='Sampling temperature (higher = more diverse)')
    parser.add_argument("--top_p", type=float, default=0.9, 
                        help='Nucleus sampling probability')

    args = parser.parse_args()

    print(f"Starting caption synthesis for {args.dataset} dataset...")
    time_start = time.time()
    main(args)
    time_end = time.time()
    print(f"\nTotal time: {time_end - time_start:.2f} seconds")