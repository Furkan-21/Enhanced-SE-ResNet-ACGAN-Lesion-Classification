"""
Training script for ACGAN on HAM10000 dataset
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import time

from models import ACGANGenerator, ACGANDiscriminator, weights_init
from utils.data import create_class_balanced_loader, get_class_distribution
from utils.visualization import save_image_grid, save_loss_plot, create_timestamp_folder, save_class_samples
from utils.metrics import calculate_fid, calculate_kid, InceptionV3Features, calculate_activation_statistics, calculate_frechet_distance

def parse_args():
    parser = argparse.ArgumentParser(description='Train ACGAN on HAM10000 dataset')
    parser.add_argument('--config', type=str, default='configs/acgan.yaml', help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='results/acgan', help='Output directory')
    parser.add_argument('--csv_file', type=str, default='data/train_split.csv', help='Path to CSV file')
    parser.add_argument('--img_dir', type=str, default='archive/hair_removed_images', help='Path to image directory')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


def train_acgan(config, args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory with timestamp
    output_dir = create_timestamp_folder(args.output_dir, prefix="acgan")
    print(f"Output directory: {output_dir}")
    
    # Create tensorboard writer
    writer = SummaryWriter(os.path.join(output_dir, 'logs'))
    
    # Save config
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump({**config, **vars(args)}, f)
    
    # Create dataloader with class balancing
    dataloader, class_weights, class_counts = create_class_balanced_loader(
        args.csv_file,
        args.img_dir,
        batch_size=config['batch_size'],
        image_size=config['image_size'],
        num_workers=config['num_workers']
    )
    
    # Print dataset info
    print(f"Number of samples: {len(dataloader.dataset)}")
    print(f"Class counts: {class_counts}")
    print(f"Class weights: {class_weights}")
    
    # Number of classes / Determine n_classes
    n_classes_from_data = len(class_counts)
    if 'n_classes' in config:
        n_classes_from_config = config['n_classes']
        print(f"Found n_classes in config: {n_classes_from_config}")
        if n_classes_from_config != n_classes_from_data:
            error_msg = (
                f"CRITICAL MISMATCH: config specifies n_classes={n_classes_from_config}, "
                f"but the loaded dataset (from {args.csv_file}) has {n_classes_from_data} classes. "
                f"Class counts found in data are: {class_counts}. "
                f"Please ensure your dataset CSV ('{args.csv_file}') is filtered for the {n_classes_from_config} target classes, "
                f"or adjust the 'n_classes' in your 'configs/acgan.yaml' file."
            )
            raise ValueError(error_msg)
        n_classes = n_classes_from_config
    else:
        n_classes = n_classes_from_data
        print(f"n_classes not in config, determined from data: {n_classes}. Ensure this is intended.")
    
    # Initialize models
    netG = ACGANGenerator(
        latent_dim=config['latent_dim'],
        n_classes=n_classes,
        channels=config['channels'],
        ngf=config['ngf']
    ).to(device)
    
    netD = ACGANDiscriminator(
        n_classes=n_classes,
        channels=config['channels'],
        ndf=config['ndf']
    ).to(device)
    
    # Print model summary
    print(f"Generator parameters: {sum(p.numel() for p in netG.parameters())}")
    print(f"Discriminator parameters: {sum(p.numel() for p in netD.parameters())}")
    
    # Initialize weights
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    # Setup optimizers
    optimizerG = optim.Adam(netG.parameters(), lr=config['lr'], betas=(config['beta1'], 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=config['lr'], betas=(config['beta1'], 0.999))
    
    # Loss functions
    adversarial_loss = nn.BCELoss()
    auxiliary_loss = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Fixed noise for visualization
    fixed_noise = torch.randn(n_classes * 8, config['latent_dim']).to(device)
    fixed_labels = torch.tensor([i for i in range(n_classes) for _ in range(8)], dtype=torch.long).to(device)
    
    # Training loop
    print("Starting training...")
    g_losses = []
    d_losses = []
    d_adv_losses = []
    d_aux_losses = []
    g_adv_losses = []
    g_aux_losses = []
    
    start_time = time.time()
    
    # For FID/KID: prepare real dataloader (subset of real images, same transform as generated images)
    from utils.data import HAM10000GanDataset
    real_dataset = HAM10000GanDataset(args.csv_file, args.img_dir, image_size=config['image_size'])
    real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=64, shuffle=True, num_workers=2)
    device_for_metrics = device
    inception = InceptionV3Features(device_for_metrics)
    # Precompute real stats
    mu_real, sigma_real = calculate_activation_statistics(real_loader, inception, device=device_for_metrics, max_samples=1000)

    for epoch in range(config['epochs']):
        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}")):
            real_images = batch['image'].to(device)
            real_labels = batch['label'].to(device)
            batch_size = real_images.size(0)
            
            # Create labels
            real_target = torch.ones(batch_size, device=device) * 0.9  # Label smoothing
            fake_target = torch.zeros(batch_size, device=device)
            
            ############################
            # (1) Update D network
            ###########################
            netD.zero_grad()
            
            # Real images
            real_pred, real_aux = netD(real_images)
            d_real_loss = adversarial_loss(real_pred, real_target)
            d_real_aux_loss = auxiliary_loss(real_aux, real_labels)
            
            # Fake images
            z = torch.randn(batch_size, config['latent_dim']).to(device)
            fake_labels = torch.randint(0, n_classes, (batch_size,)).to(device)
            fake_images = netG(z, fake_labels)
            
            fake_pred, fake_aux = netD(fake_images.detach())
            d_fake_loss = adversarial_loss(fake_pred, fake_target)
            d_fake_aux_loss = auxiliary_loss(fake_aux, fake_labels)
            
            # Total discriminator loss
            d_adv_loss = d_real_loss + d_fake_loss
            d_aux_loss = d_real_aux_loss + d_fake_aux_loss
            d_loss = d_adv_loss + config['lambda_aux'] * d_aux_loss
            
            d_loss.backward()
            optimizerD.step()
            
            ############################
            # (2) Update G network
            ###########################
            netG.zero_grad()
            
            # Generate new fake images
            z = torch.randn(batch_size, config['latent_dim']).to(device)
            fake_labels = torch.randint(0, n_classes, (batch_size,)).to(device)
            fake_images = netG(z, fake_labels)
            
            # Get discriminator predictions
            fake_pred, fake_aux = netD(fake_images)
            
            # Generator losses with increased weight for auxiliary loss
            g_adv_loss = adversarial_loss(fake_pred, real_target)
            g_aux_loss = auxiliary_loss(fake_aux, fake_labels)
            
            # Increase the weight of auxiliary loss for generator (5x the discriminator's lambda_aux)
            g_lambda_aux = config['lambda_aux'] * 5.0
            g_loss = g_adv_loss + g_lambda_aux * g_aux_loss
            
            g_loss.backward()
            optimizerG.step()
            
            # Save losses for plotting later
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            g_adv_losses.append(g_adv_loss.item())
            g_aux_losses.append(g_aux_loss.item())
            d_adv_losses.append(d_adv_loss.item())
            d_aux_losses.append(d_aux_loss.item())
            
            # Output training stats
            if i % config['log_interval'] == 0:
                elapsed = time.time() - start_time
                print(f'[{epoch+1}/{config["epochs"]}][{i}/{len(dataloader)}] '
                      f'Loss_D: {d_loss.item():.4f} Loss_G: {g_loss.item():.4f} '
                      f'D_adv: {d_adv_loss.item():.4f} D_aux: {d_aux_loss.item():.4f} '
                      f'G_adv: {g_adv_loss.item():.4f} G_aux: {g_aux_loss.item():.4f} '
                      f'Time: {elapsed:.2f}s')
                
                # Log to tensorboard
                step = epoch * len(dataloader) + i
                writer.add_scalar('Loss/Generator', g_loss.item(), step)
                writer.add_scalar('Loss/Discriminator', d_loss.item(), step)
                writer.add_scalar('Loss/G_adv', g_adv_loss.item(), step)
                writer.add_scalar('Loss/G_aux', g_aux_loss.item(), step)
                writer.add_scalar('Loss/D_adv', d_adv_loss.item(), step)
                writer.add_scalar('Loss/D_aux', d_aux_loss.item(), step)
            
            # Check how the generator is doing by saving G's output on fixed_noise
            if (i % config['sample_interval'] == 0) or ((epoch == config['epochs']-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise, fixed_labels).detach().cpu()
                    # Normalize from [-1, 1] to [0, 1]
                    fake = (fake + 1) / 2
                    
                    # Save images
                    save_image_grid(
                        fake, 
                        os.path.join(output_dir, f'samples_epoch_{epoch+1}_iter_{i}.png'),
                        nrow=8,
                        title=f'ACGAN Samples - Epoch {epoch+1}, Iteration {i}'
                    )
                    
                    # Log to tensorboard
                    img_grid = torch.clamp((fake + 1) / 2, 0, 1)
                    writer.add_images('Generated Images', img_grid, step)
        
        # Save sample images and compute metrics every 25 epochs (including epoch 0)
        if epoch % 25 == 0 or epoch == config['epochs'] - 1 or epoch == 0:
            # Generate samples for each class
            save_class_samples(
                netG, 
                device, 
                config['latent_dim'], 
                n_classes, 
                n_samples=8, 
                path=os.path.join(output_dir, f'samples_epoch_{epoch+1}')
            )
            # Compute FID/KID using 500 generated and 500 real images
            # Generate fake images
            fake_imgs = []
            fake_labels = []
            netG.eval()
            with torch.no_grad():
                for c in range(n_classes):
                    z = torch.randn(50, config['latent_dim']).to(device)
                    labels = torch.full((50,), c, dtype=torch.long).to(device)
                    gen_imgs = netG(z, labels)
                    fake_imgs.append(gen_imgs)
                    fake_labels.extend([c]*50)
            netG.train()
            fake_imgs = torch.cat(fake_imgs, dim=0)
            # Move to CPU before DataLoader!
            fake_imgs = fake_imgs.cpu()
            # Normalize to [0, 1]
            fake_imgs = (fake_imgs + 1) / 2
            # Create a DataLoader for fake images
            class FakeDataset(torch.utils.data.Dataset):
                def __init__(self, imgs):
                    self.imgs = imgs
                def __len__(self):
                    return self.imgs.shape[0]
                def __getitem__(self, idx):
                    return self.imgs[idx]  # Now always on CPU
            fake_loader = torch.utils.data.DataLoader(FakeDataset(fake_imgs), batch_size=64, shuffle=False, num_workers=2)
            # Compute FID
            mu_fake, sigma_fake = calculate_activation_statistics(fake_loader, inception, device=device_for_metrics, max_samples=350)
            fid = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
            # Compute KID (using features)
            real_features = []
            for batch in real_loader:
                images = batch['image'].to(device_for_metrics)
                features = inception(images).cpu().numpy()
                real_features.append(features)
                if len(real_features)*64 >= 350:
                    break
            real_features = np.concatenate(real_features, axis=0)[:350]
            fake_features = []
            for batch in fake_loader:
                images = batch.to(device_for_metrics)
                features = inception(images).cpu().numpy()
                fake_features.append(features)
                if len(fake_features)*64 >= 350:
                    break
            fake_features = np.concatenate(fake_features, axis=0)[:350]
            kid_mean, kid_std = calculate_kid(real_features, fake_features, subsets=10, subset_size=32)
            print(f"[Epoch {epoch+1}] FID: {fid:.2f} | KID: {kid_mean:.4f} ± {kid_std:.4f}")
        
    # Save final models only
    torch.save(netG.state_dict(), os.path.join(output_dir, 'generator_final.pth'))
    torch.save(netD.state_dict(), os.path.join(output_dir, 'discriminator_final.pth'))
    
    # Save loss plot
    save_loss_plot(g_losses, d_losses, os.path.join(output_dir, 'loss_plot.png'))
    
    # Generate and save images for classifier training with targeted augmentation
    from utils.data import save_targeted_augmentation
    classifier_aug_dir = os.path.join(output_dir, 'augmented_for_classifier')
    
    # Get class names from the dataloader
    categories = dataloader.dataset.categories
    
    # Create class mapping dictionary (class name to index)
    class_mapping = {class_name: idx for idx, class_name in enumerate(categories)}
    print(f"Class mapping for augmentation: {class_mapping}")
    
    # Use the targeted augmentation with specific counts per class
    save_targeted_augmentation(
        netG,
        device,
        config['latent_dim'],
        output_dir=classifier_aug_dir,
        class_mapping=class_mapping
    )
    print(f"Training completed. Final models and images saved to {output_dir}")
    return netG, output_dir


if __name__ == '__main__':
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Train ACGAN
    generator, output_dir = train_acgan(config, args)