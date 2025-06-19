
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

from src.datasets import PredictionDataset, UnifiedBrainDataset
from src.utils import CHECKPOINTS_PATH, DATAPATH, OUTPUT_DIR, compute_ssim_from_dataset, ensure_checkpoint_dirs, generate_and_save_predictions, normalize_image

# -------------- Argument parser setup ----------

parser = argparse.ArgumentParser(description="Train a flow matching model from T1 to T2.")
parser.add_argument('--lr_g', type=float, default=0.0002, help="Learning rate for generator")
parser.add_argument('--lr_d', type=float, default=0.00005, help="Learning rate for discriminator")
parser.add_argument('--num_workers', type=int, default=2, help="Number of workers for DataLoader")
parser.add_argument('--batchsize', type=int, default=6, help="Batch size")
parser.add_argument('--epochs', type=int, default=300, help="Number of training epochs")

args = parser.parse_args()

# ------------------ Model --------------------


class UnetSkipConnectionBlock2D(nn.Module):
    def __init__(self, outer_nc, inner_nc, in_channels=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.InstanceNorm2d, use_dropout=True):
        super(UnetSkipConnectionBlock2D, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:  # noqa
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if in_channels is None:
            in_channels = outer_nc
        downconv = nn.Conv2d(in_channels, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class UNetGenerator2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_downs=7, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=True):
        super(UNetGenerator2D, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock2D(ngf * 8, ngf * 8, in_channels=None, submodule=None, norm_layer=norm_layer, innermost=True)  # innermost
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock2D(ngf * 8, ngf * 8, in_channels=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock2D(ngf * 4, ngf * 8, in_channels=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock2D(ngf * 2, ngf * 4, in_channels=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock2D(ngf, ngf * 2, in_channels=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock2D(out_channels, ngf, in_channels=in_channels, submodule=unet_block, outermost=True, norm_layer=norm_layer)

    def forward(self, input):
        return self.model(input)


class NLayerDiscriminator2D(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        super(NLayerDiscriminator2D, self).__init__()
        if type(norm_layer) == functools.partial:  # noqa # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4  # kernel width
        padw = 1  # padding width
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

# -------------- Generation function ----------


def generate(model: DiffusionModelUNet, condition: Tensor, gen_steps: int = 20):
    device = next(model.parameters()).device
    return model(x=condition, timesteps=torch.zeros(condition.shape[0], device=device))

# ------------------ Training function ----------


def get_norm_layer():
    norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
    return norm_layer

# Initialize weights (optional, often helpful for GANs)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def train_GAN(unetG: DiffusionModelUNet, netD: NLayerDiscriminator2D, train_loader: DataLoader, val_loader: DataLoader, project: str, exp_name: str, notes: str, n_epochs: int = 200, n_epochs_decay: int = 100, lr_g: float = 0.0002, lr_d: float = 0.00005, beta1: float = 0.5, lambda_l1: float = 100.0):
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
        ensure_checkpoint_dirs()
        start_time = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    best_path_g, _ = train_GAN(
        netG=unetG,
        netD=netD,
        train_loader=train_loader,
        val_loader=val_loader,
        project="FlowMatching-Baselines",
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
        project='FlowMatching-Baselines',
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
    summary


if __name__ == "__main__":
    main()
