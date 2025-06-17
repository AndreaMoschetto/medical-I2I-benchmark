import argparse
import os
import time
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import monai
from monai.data import Dataset, list_data_collate
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    SpatialPadd,
    RandSpatialCropd,
    ResizeWithPadOrCropd, # Or use ResizeD if fixed output size is desired
    NormalizeIntensityd,
    EnsureTyped,
    MapTransform,
    Spacingd
)
from monai.utils import set_determinism

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Pix2Pix MRI to PET Translation Training")
parser.add_argument("--csv_path", type=str, default = '/cluster/project2/CU-MONDAI/Alec_Tract/mri_pet/data/adni_dataset_prepared.csv', help="Path to the CSV file with 'mri_path' and 'pet_path' columns.")
parser.add_argument("--output_dir", type=str, default="./output_pix2pix", help="Directory to save checkpoints and logs.")
parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size (Pix2Pix often uses 1 for full-resolution).")
parser.add_argument("--lr_g", type=float, default=0.0002, help="Learning rate for the generator.")
parser.add_argument("--lr_d", type=float, default=0.00005, help="Learning rate for the discriminator.")
parser.add_argument("--beta1", type=float, default=0.5, help="Adam optimizer beta1.")
parser.add_argument("--lambda_l1", type=float, default=100.0, help="Weight for L1 loss.")
parser.add_argument("--input_channels", type=int, default=1, help="Number of input channels for MRI.")
parser.add_argument("--output_channels", type=int, default=1, help="Number of output channels for PET.")
parser.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128], help="Spatial size for random cropping during training.")
parser.add_argument("--resize_size", type=int, nargs=3, default=None, help="Target size to resize images to after loading. e.g., 256 256 256")
parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
parser.add_argument("--n_epochs_decay", type=int, default=100, help="Number of epochs to start linearly decaying learning rate to zero.")
parser.add_argument("--save_epoch_freq", type=int, default=5, help="Frequency of saving checkpoints (in epochs).")
parser.add_argument("--ngf", type=int, default=64, help="Number of generator filters in the first conv layer.")
parser.add_argument("--ndf", type=int, default=64, help="Number of discriminator filters in the first conv layer.")
parser.add_argument("--no_dropout", action="store_true", help="No dropout in U-Net Generator")

args = parser.parse_args()

# For reproducibility
set_determinism(args.seed)


MRI_ZSCORE_MEAN = -49.757356053158766  # !!! REPLACE with your MRI dataset's global mean !!!
MRI_ZSCORE_STD = 24.68868751354585   # !!! REPLACE with your MRI dataset's global std !!!

PET_ZSCORE_MEAN = 0.370730756043226  # !!! REPLACE with your PET dataset's global mean !!!
PET_ZSCORE_STD = 0.667580822498126   # !!! REPLACE with your PET dataset's global std !!!

MRI_VOXEL_SPACING = (1.5, 1.5, 1.5)
PET_VOXEL_SPACING = (1.5, 1.5, 1.5) 

# --- Define U-Net Generator (3D) ---
class UNetGenerator3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_downs=7, ngf=64, norm_layer=nn.InstanceNorm3d, use_dropout=True):
        super(UNetGenerator3D, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock3D(ngf * 8, ngf * 8, in_channels=None, submodule=None, norm_layer=norm_layer, innermost=True)  # innermost
        for i in range(num_downs - 5): # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock3D(ngf * 8, ngf * 8, in_channels=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock3D(ngf * 4, ngf * 8, in_channels=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock3D(ngf * 2, ngf * 4, in_channels=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock3D(ngf, ngf * 2, in_channels=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock3D(out_channels, ngf, in_channels=in_channels, submodule=unet_block, outermost=True, norm_layer=norm_layer)

    def forward(self, input):
        return self.model(input)

class UnetSkipConnectionBlock3D(nn.Module):
    def __init__(self, outer_nc, inner_nc, in_channels=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.InstanceNorm3d, use_dropout=True):
        super(UnetSkipConnectionBlock3D, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d
        if in_channels is None:
            in_channels = outer_nc
        downconv = nn.Conv3d(in_channels, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose3d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
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


# --- Define PatchGAN Discriminator (3D) ---
class NLayerDiscriminator3D(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm3d):
        super(NLayerDiscriminator3D, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 4 # kernel width
        padw = 1 # padding width
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


# --- MONAI Dataset and Transforms ---
class PairedNiftiDataset(Dataset):
    def __init__(self, csv_file, transforms):
        all_data_df = pd.read_csv(csv_file)
        self.data_df = all_data_df[all_data_df['split'] == 'train'].reset_index(drop=True)
        self.transforms = transforms
        self.file_list = []
        for _, row in self.data_df.iterrows():
            self.file_list.append({"mri": row["scan_path"], "pet": row["suvr_path"]})

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        data_files = self.file_list[index]
        # The LoadImaged transform will load "mri" and "pet" from the paths
        return self.transforms(data_files)


# Helper for InstanceNorm used by official Pix2Pix
import functools
def get_norm_layer():

    norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)

    return norm_layer


# --- Main Training Script ---
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Transforms ---
    # Intensity scaling values (example, adjust based on your data)
    # For MRI (e.g., T1w):


    train_transforms_list = [
        LoadImaged(keys=["mri", "pet"]),
        EnsureChannelFirstd(keys=["mri", "pet"]),
        # Adjust intensity to a common range (e.g. 0-1 or -1 to 1)
        # These are examples, you MUST adjust these based on your dataset's typical intensity ranges
        NormalizeIntensityd(keys=["mri"], subtrahend=MRI_ZSCORE_MEAN, divisor=MRI_ZSCORE_STD),
        NormalizeIntensityd(keys=["pet"], subtrahend=PET_ZSCORE_MEAN, divisor=PET_ZSCORE_STD),
        # Normalize to zero mean, unit variance if preferred over scaling
        # NormalizeIntensityd(keys=["mri", "pet"], subtrahend_mean=True, divisor_std=True),
        Spacingd(keys=['mri', "pet"], pixdim=MRI_VOXEL_SPACING),
        SpatialPadd(keys=["mri", "pet"], spatial_size=(128, 160, 128), mode="constant"),
        RandSpatialCropd(keys=["mri", "pet"], roi_size=args.patch_size, random_size=False)
    ]

    train_transforms_list.extend([
        EnsureTyped(keys=["mri", "pet"], dtype=torch.float32),
    ])
    train_transforms = Compose(train_transforms_list)


    # --- Dataset and DataLoader ---
    print("Loading dataset...")
    train_ds = PairedNiftiDataset(csv_file=args.csv_path, transforms=train_transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=list_data_collate, # MONAI's default collate
        pin_memory=torch.cuda.is_available(),
    )
    print(f"Dataset loaded with {len(train_ds)} samples.")

    # --- Initialize Models ---
    norm_layer_g = get_norm_layer()
    norm_layer_d = get_norm_layer()

    netG = UNetGenerator3D(args.input_channels, args.output_channels, num_downs=7, ngf=args.ngf, norm_layer=norm_layer_g, use_dropout=not args.no_dropout).to(device)
    # For Pix2Pix, the discriminator input is MRI concatenated with either real PET or fake PET
    netD = NLayerDiscriminator3D(args.input_channels + args.output_channels, ndf=args.ndf, n_layers=3, norm_layer=norm_layer_d).to(device)

    # Initialize weights (optional, often helpful for GANs)
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    print("Initializing weights...")
    netG.apply(weights_init)
    netD.apply(weights_init)
    print("Models initialized.")

    # --- Loss Functions ---
    criterionGAN = nn.BCEWithLogitsLoss() # Sigmoid is included
    criterionL1 = nn.L1Loss()

    # --- Optimizers ---
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g, betas=(args.beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr_d, betas=(args.beta1, 0.999))

    # --- Learning Rate Schedulers ---
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 1 - (args.epochs - args.n_epochs_decay)) / float(args.n_epochs_decay + 1)
        return lr_l

    schedulerG = optim.lr_scheduler.LambdaLR(optimizerG, lr_lambda=lambda_rule)
    schedulerD = optim.lr_scheduler.LambdaLR(optimizerD, lr_lambda=lambda_rule)

    # --- Training Loop ---
    print("Starting Training Loop...")
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        netG.train()
        netD.train()

        epoch_loss_g = 0
        epoch_loss_d = 0
        epoch_loss_g_gan = 0
        epoch_loss_g_l1 = 0

        for i, batch_data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            real_A = batch_data["mri"].to(device) # MRI
            real_B = batch_data["pet"].to(device) # PET

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizerD.zero_grad()

            # Real images
            # Discriminator input: concatenate MRI (real_A) and real PET (real_B)
            real_AB = torch.cat((real_A, real_B), 1)
            pred_real = netD(real_AB)
            # Label smoothing: use 0.9 for real instead of 1.0
            target_real = torch.full(pred_real.shape, 0.9 if torch.rand(1).item() > 0.05 else 1.0, device=device, dtype=torch.float32) # Small chance of flipping for robustness
            loss_D_real = criterionGAN(pred_real, target_real)

            # Fake images
            fake_B = netG(real_A).detach() # Detach to avoid backprop to G here
            # Discriminator input: concatenate MRI (real_A) and fake PET (fake_B)
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

            # Generate fake PET
            fake_B_for_G = netG(real_A)
            # Discriminator input for G's adversarial loss
            fake_AB_for_G = torch.cat((real_A, fake_B_for_G), 1)
            pred_fake_G = netD(fake_AB_for_G)
            # Generator wants discriminator to think fake images are real
            target_real_for_G = torch.ones_like(pred_fake_G, device=device, dtype=torch.float32) # No smoothing for G's target
            loss_G_GAN = criterionGAN(pred_fake_G, target_real_for_G)

            # L1 loss (reconstruction loss)
            loss_G_L1 = criterionL1(fake_B_for_G, real_B) * args.lambda_l1

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

        # --- Logging ---
        avg_loss_d = epoch_loss_d / len(train_loader)
        avg_loss_g = epoch_loss_g / len(train_loader)
        avg_loss_g_gan = epoch_loss_g_gan / len(train_loader)
        avg_loss_g_l1 = epoch_loss_g_l1 / len(train_loader)

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Time: {time.time() - epoch_start_time:.2f}s "
              f"Loss_D: {avg_loss_d:.4f} "
              f"Loss_G: {avg_loss_g:.4f} (GAN: {avg_loss_g_gan:.4f}, L1: {avg_loss_g_l1:.4f}) "
              f"LR_G: {optimizerG.param_groups[0]['lr']:.6f} "
              f"LR_D: {optimizerD.param_groups[0]['lr']:.6f}")

        # --- Save Checkpoints ---
        if (epoch + 1) % args.save_epoch_freq == 0 or (epoch + 1) == args.epochs:
            torch.save(netG.state_dict(), os.path.join(args.output_dir, f"netG_epoch_{epoch+1}.pth"))
            torch.save(netD.state_dict(), os.path.join(args.output_dir, f"netD_epoch_{epoch+1}.pth"))
            print(f"Saved checkpoints for epoch {epoch+1}")

    print("Training finished.")

if __name__ == "__main__":

    args = parser.parse_args()

    main(args)