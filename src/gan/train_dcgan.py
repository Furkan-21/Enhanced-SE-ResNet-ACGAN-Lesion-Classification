"""
Training script for DCGAN on a single class of HAM10000 dataset
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

from models import DCGANGenerator, DCGANDiscriminator, weights_init
from utils.data import create_single_class_loader, get_class_distribution
from utils.visualization import save_image_grid, save_loss_plot, create_timestamp_folder, save_dcgan_samples


def parse_args():
    parser = argparse.ArgumentParser(description='Train DCGAN on HAM10000 dataset')
    parser.add_argument('--config', type=str, default='configs/dcgan.yaml', help='Path to config file')
    parser.add_argument('--class_name', type=str, required=True, help='Class to train on (e.g., "VASC", "DF")')
    parser.add_argument('--output_dir', type=str, default='results/dcgan', help='Output directory')
    parser.add_argument('--csv_file', type=str, default='data/train_split.csv', help='Path to CSV file')
    parser.add_argument('--img_dir', type=str, default='archive/hair_removed_images', help='Path to image directory')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


def train_dcgan(config, args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory with timestamp
    output_dir = create_timestamp_folder(args.output_dir, prefix=f"dcgan_{args.class_name.lower()}")
    print(f"Output directory: {output_dir}")
    
    # Create tensorboard writer
    writer = SummaryWriter(os.path.join(output_dir, 'logs'))
    
    # Save config
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump({**config, **vars(args)}, f)
    
    # Create dataloader
    dataloader = create_single_class_loader(
        args.csv_file,
        args.img_dir,
        args.class_name,
        batch_size=config['batch_size'],
        image_size=config['image_size'],
        num_workers=config['num_workers']
    )
    
    # Print dataset info
    print(f"Training on class: {args.class_name}")
    print(f"Number of samples: {len(dataloader.dataset)}")
    
    # Initialize models
    netG = DCGANGenerator(
        latent_dim=config['latent_dim'],
        channels=config['channels'],
        ngf=config['ngf']
    ).to(device)
    
    netD = DCGANDiscriminator(
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
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Fixed noise for visualization
    fixed_noise = torch.randn(64, config['latent_dim']).to(device)
    
    # Training loop
    print("Starting training...")
    g_losses = []
    d_losses = []
    
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}")):
            ############################
            # (1) Update D network
            ###########################
            # Train with real batch
            netD.zero_grad()
            real_images = batch['image'].to(device)
            batch_size = real_images.size(0)
            
            # Use label smoothing for real labels (0.9 instead of 1.0)
            real_label = 0.9
            fake_label = 0.0
            
            # Forward pass real batch through D
            output = netD(real_images)
            
            # Calculate loss on real batch
            real_labels = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            errD_real = criterion(output, real_labels)
            errD_real.backward()
            D_x = output.mean().item()
            
            # Train with fake batch
            # Generate batch of latent vectors
            noise = torch.randn(batch_size, config['latent_dim']).to(device)
            
            # Generate fake images
            fake = netG(noise)
            
            # Classify fake batch with D
            output = netD(fake.detach())
            
            # Calculate D's loss on fake batch
            fake_labels = torch.full((batch_size,), fake_label, dtype=torch.float, device=device)
            errD_fake = criterion(output, fake_labels)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            
            # Add the gradients from the real and fake batches
            errD = errD_real + errD_fake
            
            # Update D
            optimizerD.step()
            
            ############################
            # (2) Update G network
            ###########################
            netG.zero_grad()
            
            # Since we just updated D, perform another forward pass of fake batch through D
            output = netD(fake)
            
            # Calculate G's loss based on this output
            # We want the generator to generate images that the discriminator thinks are real
            labels = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            errG = criterion(output, labels)
            errG.backward()
            D_G_z2 = output.mean().item()
            
            # Update G
            optimizerG.step()
            
            # Save losses for plotting later
            g_losses.append(errG.item())
            d_losses.append(errD.item())
            
            # Output training stats
            if i % config['log_interval'] == 0:
                elapsed = time.time() - start_time
                print(f'[{epoch+1}/{config["epochs"]}][{i}/{len(dataloader)}] '
                      f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                      f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f} '
                      f'Time: {elapsed:.2f}s')
                
                # Log to tensorboard
                step = epoch * len(dataloader) + i
                writer.add_scalar('Loss/Generator', errG.item(), step)
                writer.add_scalar('Loss/Discriminator', errD.item(), step)
                writer.add_scalar('D_x', D_x, step)
                writer.add_scalar('D_G_z/D_G_z1', D_G_z1, step)
                writer.add_scalar('D_G_z/D_G_z2', D_G_z2, step)
            
            # Check how the generator is doing by saving G's output on fixed_noise
            if (i % config['sample_interval'] == 0) or ((epoch == config['epochs']-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                    # Normalize from [-1, 1] to [0, 1]
                    fake = (fake + 1) / 2
                    
                    # Save images
                    save_image_grid(
                        fake, 
                        os.path.join(output_dir, f'samples_epoch_{epoch+1}_iter_{i}.png'),
                        nrow=8,
                        title=f'DCGAN Samples - Epoch {epoch+1}, Iteration {i}'
                    )
                    
                    # Log to tensorboard
                    img_grid = torch.clamp((fake + 1) / 2, 0, 1)
                    writer.add_images('Generated Images', img_grid, step)
        
        # Save models after each epoch
        if (epoch + 1) % config['save_interval'] == 0 or epoch == config['epochs'] - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': netG.state_dict(),
                'optimizer_state_dict': optimizerG.state_dict(),
                'loss': errG.item(),
            }, os.path.join(output_dir, f'generator_epoch_{epoch+1}.pth'))
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': netD.state_dict(),
                'optimizer_state_dict': optimizerD.state_dict(),
                'loss': errD.item(),
            }, os.path.join(output_dir, f'discriminator_epoch_{epoch+1}.pth'))
    
    # Save final models
    torch.save(netG.state_dict(), os.path.join(output_dir, 'generator_final.pth'))
    torch.save(netD.state_dict(), os.path.join(output_dir, 'discriminator_final.pth'))
    
    # Save loss plot
    save_loss_plot(g_losses, d_losses, os.path.join(output_dir, 'loss_plot.png'))
    
    # Generate final samples
    save_dcgan_samples(
        netG, 
        device, 
        config['latent_dim'], 
        n_samples=100, 
        path=os.path.join(output_dir, 'final_samples.png')
    )
    
    print(f"Training completed. Results saved to {output_dir}")
    return netG, output_dir


if __name__ == '__main__':
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Train DCGAN
    generator, output_dir = train_dcgan(config, args)
