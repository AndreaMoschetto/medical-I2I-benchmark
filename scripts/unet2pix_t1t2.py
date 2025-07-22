
import argparse
import functools
import os
import time

import torch
import torch.optim as optim
from monai.data import DataLoader

from generative.networks.nets import DiffusionModelUNet

from torchvision import transforms
import torchvision
import wandb
from torch import Tensor, nn
from tqdm import trange

from t1t2converter.datasets import PredictionDataset, UnifiedBrainDataset
from t1t2converter.utils import CHECKPOINTS_PATH, DATAPATH, OUTPUT_DIR, compute_ssim_from_dataset, generate_and_save_predictions, normalize_image, weights_init
from t1t2converter.models import NLayerDiscriminator2D
# -------------- Argument parser setup ----------

parser = argparse.ArgumentParser(description="Train a flow matching model from T1 to T2.")
parser.add_argument('--lr_g', type=float, default=0.0002, help="Learning rate for generator")
parser.add_argument('--lr_d', type=float, default=0.00005, help="Learning rate for discriminator")
parser.add_argument('--num_workers', type=int, default=2, help="Number of workers for DataLoader")
parser.add_argument('--batchsize', type=int, default=6, help="Batch size")
parser.add_argument('--epochs', type=int, default=300, help="Number of training epochs")

args = parser.parse_args()


# -------------- Generation function ----------
def generate(model: DiffusionModelUNet, condition: Tensor, gen_steps: int = 20):
    device = next(model.parameters()).device
    return model(x=condition, timesteps=torch.zeros(condition.shape[0], device=device))

# ------------------ Training function ----------


def train_GAN(unetG: DiffusionModelUNet, netD: NLayerDiscriminator2D, device: str, train_loader: DataLoader, val_loader: DataLoader, project: str, exp_name: str, notes: str, n_epochs: int = 200, n_epochs_decay: int = 100, lr_g: float = 0.0002, lr_d: float = 0.00005, beta1: float = 0.5, lambda_l1: float = 100.0):
    with wandb.init(
        project=project,
        name=exp_name,
        notes=notes,
        tags=["flow", "brain", "diffusion"],
        config={
            'modelG': unetG.__class__.__name__,
            'modelD': netD.__class__.__name__,
            'epochs': n_epochs,
            'n_epochs_decay': n_epochs_decay,
            'batch_size': train_loader.batch_size,
            'num_workers': train_loader.num_workers,
            'optimizer': 'Adam',
            'learning_rate_g': lr_g,
            'learning_rate_d': lr_d,
            'beta1': beta1,
            'lambda_l1': lambda_l1,
            'loss_functions': 'BCEWithLogitsLoss, L1Loss',
            'device': str(torch.cuda.get_device_name(0)
                          if torch.cuda.is_available() else "CPU"),
        }
    ) as run:
        start_time = time.time()
        print(f"Using device: {device}")
        # --- Models ---
        unetG = unetG.to(device)
        netD = netD.to(device)
        print("Initializing weights...")
        # unetG.apply(weights_init)
        netD.apply(weights_init)
        print("Models initialized.")

        # --- Loss Functions ---
        criterionGAN = nn.BCEWithLogitsLoss()  # Sigmoid is included
        criterionL1 = nn.L1Loss()
        # --- Optimizers ---
        optimizerG = optim.Adam(unetG.parameters(), lr=lr_g, betas=(beta1, 0.999))
        optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, 0.999))
        # --- Learning Rate Schedulers ---

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - (n_epochs - n_epochs_decay)) / float(n_epochs_decay + 1)
            return lr_l
        schedulerG = optim.lr_scheduler.LambdaLR(optimizerG, lr_lambda=lambda_rule)
        schedulerD = optim.lr_scheduler.LambdaLR(optimizerD, lr_lambda=lambda_rule)

        # --- Training Loop ---
        print("Starting Training Loop...")
        best_val_g_l1_loss = float('inf')
        best_path_g = None
        best_path_d = None
        for epoch in trange(n_epochs, desc="Epochs"):
            epoch_start_time = time.time()
            unetG.train()
            netD.train()

            epoch_loss_g = 0
            epoch_loss_d = 0
            epoch_loss_g_gan = 0
            epoch_loss_g_l1 = 0
            for i, batch_data in enumerate(train_loader):
                real_A = batch_data['t1'].to(device)
                real_B = batch_data['t2'].to(device)
                # ---------------------
                #  Train Discriminator
                # ---------------------
                optimizerD.zero_grad()
                # Real images
                # Discriminator input: concatenate T1 (real_A) and real T2 (real_B)
                real_AB = torch.cat((real_A, real_B), 1)
                pred_real = netD(real_AB)
                # Label smoothing: use 0.9 for real instead of 1.0
                target_real = torch.full(pred_real.shape, 0.9 if torch.rand(1).item() > 0.05 else 1.0, device=device, dtype=torch.float32)  # Small chance of flipping for robustness
                loss_D_real = criterionGAN(pred_real, target_real)

                # Fake images
                # fake_B = netG(real_A).detach() # Detach to avoid backprop to G here
                fake_B = unetG(x=real_A, timesteps=torch.zeros(real_A.shape[0], device=device)).detach()  # Detach to avoid backprop to G here

                # Discriminator input: concatenate T1 (real_A) and fake T2 (fake_B)
                fake_AB = torch.cat((real_A, fake_B), 1)
                pred_fake = netD(fake_AB)
                # Label smoothing: use 0.1 for fake instead of 0.0
                target_fake = torch.full(pred_fake.shape, 0.1 if torch.rand(1).item() > 0.05 else 0.0, device=device, dtype=torch.float32)
                loss_D_fake = criterionGAN(pred_fake, target_fake)

                # Total discriminator loss
                loss_D = (loss_D_real + loss_D_fake) * 0.5
                loss_D.backward()
                optimizerD.step()

                epoch_loss_d += loss_D.item()

                # -----------------
                #  Train Generator
                # -----------------
                optimizerG.zero_grad()

                # Generate fake T2
                fake_B_for_G = unetG(x=real_A, timesteps=torch.zeros(real_A.shape[0], device=device))
                # Discriminator input for G's adversarial loss
                fake_AB_for_G = torch.cat((real_A, fake_B_for_G), 1)
                pred_fake_G = netD(fake_AB_for_G)
                # Generator wants discriminator to think fake images are real
                target_real_for_G = torch.ones_like(pred_fake_G, device=device, dtype=torch.float32)  # No smoothing for G's target
                loss_G_GAN = criterionGAN(pred_fake_G, target_real_for_G)

                # L1 loss (reconstruction loss)
                loss_G_L1 = criterionL1(fake_B_for_G, real_B) * lambda_l1

                # Total generator loss
                loss_G = loss_G_GAN + loss_G_L1
                loss_G.backward()
                optimizerG.step()

                epoch_loss_g += loss_G.item()
                epoch_loss_g_gan += loss_G_GAN.item()
                epoch_loss_g_l1 += loss_G_L1.item()

            # Update learning rates
            schedulerG.step()
            schedulerD.step()

            # --- Training Losses ---
            avg_loss_d = epoch_loss_d / len(train_loader)
            avg_loss_g = epoch_loss_g / len(train_loader)
            avg_loss_g_gan = epoch_loss_g_gan / len(train_loader)
            avg_loss_g_l1 = epoch_loss_g_l1 / len(train_loader)

            # -----------------------
            #  Validation Phase
            # -----------------------
            unetG.eval()
            netD.eval()
            val_loss_g = 0
            val_loss_d = 0
            val_loss_g_gan = 0
            val_loss_g_l1 = 0

            with torch.no_grad():
                for val_batch in val_loader:
                    real_A = val_batch['t1'].to(device)
                    real_B = val_batch['t2'].to(device)

                    # --- Discriminator ---
                    real_AB = torch.cat((real_A, real_B), 1)
                    pred_real = netD(real_AB)
                    target_real = torch.ones_like(pred_real, device=device)
                    loss_D_real = criterionGAN(pred_real, target_real)

                    fake_B = unetG(x=real_A, timesteps=torch.zeros(real_A.shape[0], device=device))
                    fake_AB = torch.cat((real_A, fake_B), 1)
                    pred_fake = netD(fake_AB)
                    target_fake = torch.zeros_like(pred_fake, device=device)
                    loss_D_fake = criterionGAN(pred_fake, target_fake)

                    loss_D = (loss_D_real + loss_D_fake) * 0.5
                    val_loss_d += loss_D.item()

                    # --- Generator ---
                    pred_fake_G = netD(fake_AB)
                    target_real_for_G = torch.ones_like(pred_fake_G, device=device)
                    loss_G_GAN = criterionGAN(pred_fake_G, target_real_for_G)
                    loss_G_L1 = criterionL1(fake_B, real_B) * lambda_l1
                    loss_G = loss_G_GAN + loss_G_L1

                    val_loss_g += loss_G.item()
                    val_loss_g_gan += loss_G_GAN.item()
                    val_loss_g_l1 += loss_G_L1.item()

            # --- Average Validation Losses ---
            avg_val_loss_d = val_loss_d / len(val_loader)
            avg_val_loss_g = val_loss_g / len(val_loader)
            avg_val_loss_g_gan = val_loss_g_gan / len(val_loader)
            avg_val_loss_g_l1 = val_loss_g_l1 / len(val_loader)

            epoch_duration = time.time() - epoch_start_time

            run.log(
                {
                    "epoch": epoch + 1,
                    "loss_G": avg_loss_g,
                    "loss_G_GAN": avg_loss_g_gan,
                    "loss_G_L1": avg_loss_g_l1,
                    "loss_D": avg_loss_d,
                    "val_loss_G": avg_val_loss_g,
                    "val_loss_G_GAN": avg_val_loss_g_gan,
                    "val_loss_G_L1": avg_val_loss_g_l1,
                    "val_loss_D": avg_val_loss_d,
                    "lr_G": optimizerG.param_groups[0]['lr'],
                    "lr_D": optimizerD.param_groups[0]['lr'],
                    "epoch_time_minutes": epoch_duration // 60
                }
            )
            if epoch % 5 == 0 or (epoch + 1) == n_epochs or val_loss_g_l1 < best_val_g_l1_loss:
                # Log sample images
                with torch.no_grad():
                    sample_batch = next(iter(val_loader))
                    sample_real_A = sample_batch['t1'][0].unsqueeze(0).to(device)
                    sample_real_B = sample_batch['t2'][0].to(device)
                    sample_fake_B = unetG(x=real_A, timesteps=torch.zeros(real_A.shape[0], device=device))
                    imgs = torch.stack([sample_real_A.squeeze(0),
                                        sample_real_B,
                                        normalize_image(sample_fake_B.squeeze(0))], dim=0)
                    grid = torchvision.utils.make_grid(imgs, nrow=3, scale_each=True)
                    run.log({'each5e_generation': wandb.Image(grid, caption=f'Epoch {epoch + 1}')})

                if val_loss_g_l1 < best_val_g_l1_loss:
                    # Save best models
                    best_val_g_l1_loss = val_loss_g_l1
                    path_g = f'{CHECKPOINTS_PATH}/checkpoint_{exp_name}_{epoch+1}__generator_best.pth'
                    path_d = f'{CHECKPOINTS_PATH}/checkpoint_{exp_name}_{epoch+1}__discriminator_best_g.pth'  # Note: discriminato is the one that was trained with the best generator
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': unetG.state_dict(),
                        'optimizer_state_dict': optimizerG.state_dict(),
                    }, path_g)
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': netD.state_dict(),
                        'optimizer_state_dict': optimizerD.state_dict(),
                    }, path_d)
                    if best_path_g is not None and os.path.exists(best_path_g):
                        os.remove(best_path_g)
                    if best_path_d is not None and os.path.exists(best_path_d):
                        os.remove(best_path_d)
                    best_path_g = path_g
                    best_path_d = path_d
                else:
                    # Save backups every 5 epochs
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': unetG.state_dict(),
                        'optimizer_state_dict': optimizerG.state_dict(),
                    }, f'{CHECKPOINTS_PATH}/backups/checkpoint_{exp_name}_{epoch+1}__generator.pth')
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': netD.state_dict(),
                        'optimizer_state_dict': optimizerD.state_dict(),
                    }, f'{CHECKPOINTS_PATH}/backups/checkpoint_{exp_name}_{epoch+1}__discriminator.pth')
        run.log({"total_running_hours": (time.time() - start_time) // 3600})
        return best_path_g, best_path_d


def main():
    # ---------- Dataset and Model setup ----------
    transform = transforms.Compose([
        transforms.Pad(padding=(5, 3, 5, 3), fill=0),
        transforms.ToTensor(),  # Normalize to [0, 1]
    ])

    train_dataset = UnifiedBrainDataset(root_dir=DATAPATH, transform=transform, split="train")
    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, num_workers=args.num_workers, shuffle=True)
    val_dataset = UnifiedBrainDataset(root_dir=DATAPATH, transform=transform, split="val")
    val_loader = DataLoader(val_dataset, batch_size=args.batchsize, num_workers=args.num_workers, shuffle=False)
    test_dataset = UnifiedBrainDataset(root_dir=DATAPATH, transform=transform, split="test")
    test_loader = DataLoader(test_dataset, batch_size=args.batchsize, num_workers=args.num_workers, shuffle=False)

    def get_norm_layer():
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
        return norm_layer

    norm_layer_d = get_norm_layer()

    unetG = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1
    )

    netD = NLayerDiscriminator2D(
        input_nc=2,
        ndf=64,
        n_layers=3,
        norm_layer=norm_layer_d
    )

    # ---------- Model training ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_name = f"pix2pix-t1t2-brain{args.epochs}e"
    prediction_dir = f"{OUTPUT_DIR}/predictions/{exp_name}"
    project_name = 'FlowMatching-Baselines'

    best_path_g, _ = train_GAN(
        netG=unetG,
        netD=netD,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        project=project_name,
        exp_name=exp_name,
        notes="Baseline Unet2Pix for T1-T2 conversion",
        n_epochs=args.epochs,
        n_epochs_decay=100,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        beta1=0.5,
        lambda_l1=100.0
    )

    # Load the best checkpoint
    checkpoint = torch.load(best_path_g, map_location=device)
    unetG.load_state_dict(checkpoint['model_state_dict'])
    unetG.to(device)

    # ---------- Model evaluation and prediction generation ----------
    with wandb.init(
        project=project_name,
        name=f'evaluation-{exp_name}',
        notes="Evaluation of the diffusion model on the test set.",
    ) as run:
        generate_and_save_predictions(
            model=unetG,
            test_loader=test_loader,
            device=device,
            output_dir=prediction_dir,
            generation_f=generate,
            wandb_run=run,
            just_one_batch=False)

        out_dataset = PredictionDataset(prediction_dir)
        summary = compute_ssim_from_dataset(out_dataset, wandb_run=run)
        print(summary)


if __name__ == "__main__":
    main()
