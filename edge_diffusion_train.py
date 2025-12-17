"""EDGE-VLDD Diffusion Training Script

This module implements the training pipeline for EDGE-VLDD using Stable Diffusion
with contrastive and diversity losses for vision-language dataset distillation.
"""

import sys
import time

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

from diffusers import StableDiffusionPipeline
from transformers import CLIPModel

sys.path.append('./evaluation')
from dataset import get_dataset_flickr

class ContrastiveLoss(nn.Module):
    """Contrastive loss for vision-language alignment.
    
    Computes bidirectional contrastive loss between image and text features,
    encouraging matching pairs to have high similarity.
    
    Args:
        temperature: Scaling factor for logits (default: 0.1)
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, image_features, text_features):
        """Compute contrastive loss.
        
        Args:
            image_features: Image embeddings [batch_size, dim]
            text_features: Text embeddings [batch_size, dim]
            
        Returns:
            Averaged bidirectional contrastive loss
        """
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        logits_per_image = torch.matmul(image_features, text_features.T) / self.temperature
        logits_per_text = torch.matmul(text_features, image_features.T) / self.temperature
        labels = torch.arange(logits_per_image.shape[0], device=image_features.device)
        return 0.5 * (self.loss_fn(logits_per_image, labels) +
                      self.loss_fn(logits_per_text, labels))

class DiversityLoss(nn.Module):
    """Diversity loss to encourage variety in the generated samples.
    
    Computes pairwise similarity between joint vision-language embeddings
    and penalizes high similarity to promote diversity.
    """
    def __init__(self):
        super().__init__()

    def forward(self, image_emd, text_emd):
        """Compute diversity loss.
        
        Args:
            image_emd: Image embeddings [batch_size, dim]
            text_emd: Text embeddings [batch_size, dim]
            
        Returns:
            Average pairwise similarity (lower is more diverse)
        """
        joint_emd = torch.cat([image_emd, text_emd], dim=-1)
        joint_emd = F.normalize(joint_emd, p=2, dim=-1)
        sim = torch.matmul(joint_emd, joint_emd.T)
        mask = torch.eye(sim.size(0), device=sim.device).bool()
        sim = sim.masked_fill(mask, 0)
        return sim.sum() / (sim.size(0) * (sim.size(0) - 1))

def mark_difffit_trainable(model, is_bitfit=False):
    trainable_names = ['bias'] if is_bitfit else ['bias', 'norm', 'gamma', 'y_embed']
    for n, p in model.named_parameters():
        p.requires_grad = any(kw in n for kw in trainable_names)
    return model

def count_trainable_parameters(module, name):
    params = [p for p in module.parameters() if p.requires_grad]
    total = sum(p.numel() for p in params)
    print(f"Trainable params in {name}: {total/1e6:.2f} M")
    return params

def main(args):
    """Main training function for EDGE-VLDD.
    
    Args:
        args: Argument namespace containing training configuration
    """
    # Setup data transforms
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
    ])

    # Load dataset
    trainloader, testloader, *_ = get_dataset_flickr(args, finetune=True, fraction=args.fraction)

    # Load Stable Diffusion pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "checkpoints/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
    ).to("cuda")

    # Initialize loss functions
    contrastive_loss_fn = ContrastiveLoss(temperature=0.1)
    diversity_loss_fn = DiversityLoss()

    # Configure trainable parameters
    for p in pipe.vae.parameters():
        p.requires_grad = False
    _ = count_trainable_parameters(pipe.vae, "pipe.vae")

    pipe.unet = mark_difffit_trainable(pipe.unet, is_bitfit=False)
    opt_params = count_trainable_parameters(pipe.unet, "pipe.unet")

    for p in pipe.text_encoder.parameters():
        p.requires_grad = False
    _ = count_trainable_parameters(pipe.text_encoder, "pipe.text_encoder")

    optimizer = optim.SGD(opt_params, lr=args.lr, weight_decay=0)

    # Load CLIP model for feature extraction
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda").eval()

    # Training loop
    for epoch in range(args.num_epochs):
        for images, captions, _ in tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            # Prepare inputs
            images = images.half().cuda()
            text_inputs = pipe.tokenizer(captions, padding="max_length", return_tensors="pt", truncation=True).to("cuda")
            text_embeddings = pipe.text_encoder(text_inputs.input_ids)[0]

            optimizer.zero_grad()

            # Forward diffusion process
            latents = pipe.vae.encode(images).latent_dist.sample() * 0.18215
            bsz = latents.size(0)
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            noise = torch.randn_like(latents)
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
            
            # Predict noise
            noise_pred = pipe.unet(latents, timesteps, encoder_hidden_states=text_embeddings).sample

            # Denoise to recover clean latents
            alpha_prod_t = pipe.scheduler.alphas_cumprod[timesteps]
            beta_prod_t = 1 - alpha_prod_t
            alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)
            beta_prod_t = beta_prod_t.view(-1, 1, 1, 1)
            denoised_latents = (noisy_latents - beta_prod_t.sqrt() * noise_pred) / alpha_prod_t.sqrt()
            denoised_latents = denoised_latents.half()

            # Decode latents and extract features
            image_features = pipe.vae.decode(denoised_latents).sample.float()
            image_proj = clip_model.get_image_features(image_features)
            text_proj = clip_model.get_text_features(**text_inputs)

            # Compute losses
            contrastive_loss = contrastive_loss_fn(image_proj, text_proj)
            diversity_loss = diversity_loss_fn(image_proj, text_proj)
            loss = args.contrastive_lambda * contrastive_loss + args.diversity_lambda * diversity_loss

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

    # Save trained model
    pipe.save_pretrained(args.checkpoint_path)
    print(f"Model saved to {args.checkpoint_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='EDGE-VLDD: Training vision-language dataset distillation with Stable Diffusion')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='flickr30k', help='Dataset name')
    parser.add_argument('--image_root', type=str, default='distill_utils/data/Flickr30k/', help='Path to image root directory')
    parser.add_argument('--ann_root', type=str, default='./data/Flickr30k_ann/', help='Path to annotation root directory')
    parser.add_argument('--batch_size_train', type=int, default=128, help='Training batch size')
    parser.add_argument('--batch_size_test', type=int, default=128, help='Test batch size')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size')
    parser.add_argument('--no_aug', action='store_true', default=False, help='Disable data augmentation')

    # Model arguments
    parser.add_argument('--image_encoder', type=str, default='nfnet', help='Image encoder architecture')
    parser.add_argument('--text_encoder', type=str, default='bert', choices=['bert', 'clip', 'distilbert'], help='Text encoder architecture')
    parser.add_argument('--only_has_image_projection', type=bool, default=False, help='Use only image projection')
    parser.add_argument('--distill', type=bool, default=True, help='Enable distillation mode')
    parser.add_argument('--text_pretrained', type=bool, default=True, help='Use pretrained text encoder')
    parser.add_argument('--image_pretrained', type=bool, default=True, help='Use pretrained image encoder')
    parser.add_argument('--text_trainable', type=bool, default=False, help='Make text encoder trainable')
    parser.add_argument('--image_trainable', type=bool, default=True, help='Make image encoder trainable')

    # Training arguments
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--contrastive_lambda', type=float, default=1.0, help='Weight for contrastive loss')
    parser.add_argument('--diversity_lambda', type=float, default=1.0, help='Weight for diversity loss')
    parser.add_argument('--fraction', type=float, default=1.0, help='Fraction of dataset to use')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/pipe', help='Path to save model checkpoints')

    args = parser.parse_args()
    args.distill_image = args.image_root
    args.distill_ann = args.ann_root

    start = time.time()
    main(args)
    print(f"Training completed in {time.time() - start:.1f} s")
