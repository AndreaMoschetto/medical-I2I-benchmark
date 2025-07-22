import os
import argparse

import torch
import time
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import Tensor
from generative.inferers import DiffusionInferer, ControlNetDiffusionInferer
from generative.networks.nets import DiffusionModelUNet, ControlNet
from generative.networks.schedulers import DDPMScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb
from tqdm import trange

from t1t2converter.datasets import PredictionDataset, UnifiedBrainDataset
from t1t2converter.utils import CHECKPOINTS_PATH, DATAPATH, OUTPUT_DIR, compute_ssim_from_dataset, generate_and_save_predictions, normalize_image

# -------------- Argument parser setup ----------

parser = argparse.ArgumentParser(description="Train a controlnet model from T1 to T2.")
parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate")
parser.add_argument('--lr_min', type=float, default=1e-6, help="Minimum learning rate for CosineAnnealingLR")
parser.add_argument('--num_workers', type=int, default=2, help="Number of workers for DataLoader")
parser.add_argument('--batchsize', type=int, default=6, help="Batch size")
parser.add_argument('--epochs', type=int, default=300, help="Number of training epochs")
parser.add_argument('--df_model', type=str, default=f'{CHECKPOINTS_PATH}/checkpoint_diffusion-t2-brain300e_164_best.pth', help="Path to the pretrained diffusion model checkpoint")

args = parser.parse_args()

# -------------- Generation function ----------


def make_generate_fn(df_model: DiffusionModelUNet, inferer: DiffusionInferer):
    @torch.no_grad()
    def generate_wrapped(model: ControlNet, condition: Tensor, gen_steps: int = 20):
        device = next(model.parameters()).device
        noise = torch.randn_like(condition).to(device)
        return inferer.sample(input_noise=noise, diffusion_model=df_model, controlnet=model, scheduler=inferer.scheduler, cn_cond=condition, verbose=False)
    return generate_wrapped
# ------------------ Training function ----------


def train_controlnet(cn_model: ControlNet, model: DiffusionModelUNet, device: str, inferer: ControlNetDiffusionInferer, train_loader: DataLoader, val_loader: DataLoader, project: str, exp_name: str, notes: str, n_epochs: int = 10, lr: float = 1e-3, generation_steps: int = 100):
    with wandb.init(
        project=project,
        name=exp_name,
        notes=notes,
        tags=["flow", "brain", "diffusion"],
        config={
            'model': model.__class__.__name__,
            'epochs': n_epochs,
            'batch_size': train_loader.batch_size,
            'num_workers': train_loader.num_workers,
            'optimizer': 'Adam',
            'learning_rate': lr,
            'loss_function': 'MSELoss',
            'generation_steps': generation_steps,
            'device': str(torch.cuda.get_device_name(0)
                          if torch.cuda.is_available() else "CPU"),
        }
    ) as run:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

        cn_model.to(device)
        cn_model.train()

        optimizer = torch.optim.Adam(cn_model.parameters(), lr=lr)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        best_model_path = None
        start_time = time.time()
        for e in trange(n_epochs, desc="Epochs"):
            start_e_time = time.time()
            # Training
            cn_model.train()
            train_losses = []
            for batch in train_loader:
                t2_targets = batch["t2"].to(device)  # [B, 1, H, W]
                t1_cond = batch["t1"].to(device)  # [B, 1, H, W]
                noise = torch.randn_like(t2_targets).to(device)

                B = t2_targets.shape[0]
                # Create timesteps
                timesteps = torch.randint(0, generation_steps, (B,), device=t2_targets.device).long()

                # Get model prediction
                optimizer.zero_grad()
                noise_pred = inferer(inputs=t2_targets, diffusion_model=model, controlnet=cn_model, noise=noise, timesteps=timesteps, cn_cond=t1_cond)
                loss = criterion(noise_pred.float(), noise.float())
                train_losses.append(loss.item())
                loss.backward()
                optimizer.step()
            run.log({"train_loss": sum(train_losses) / len(train_losses)})

            # Validation
            cn_model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    t2_targets = batch["t2"].to(device)
                    t1_cond = batch["t1"].to(device)
                    noise = torch.randn_like(t2_targets).to(device)
                    B = t2_targets.shape[0]
                    timesteps = torch.randint(0, generation_steps, (B,), device=t2_targets.device).long()
                    noise_pred = inferer(inputs=t2_targets, diffusion_model=model, controlnet=cn_model, noise=noise, timesteps=timesteps, cn_cond=t1_cond)
                    val_loss = criterion(noise_pred.float(), noise.float())
                    val_losses.append(val_loss.item())
            batch_val_loss = sum(val_losses) / len(val_losses)
            run.log({"val_loss": batch_val_loss})
            lr_scheduler.step()
            run.log({"lr": lr_scheduler.get_last_lr()[0]})
            e_time = time.time() - start_e_time
            run.log({"epoch_time_minutes": e_time // 60})

            # Checkpoint
            if e % 5 == 0 or e == n_epochs - 1 or batch_val_loss < best_val_loss:
                sample_batch = next(iter(val_loader))  # just one batch

                with torch.no_grad():
                    t2_target = sample_batch["t2"][0].unsqueeze(0).to(device)
                    t1_cond = sample_batch["t1"][0].unsqueeze(0).to(device)
                    noise = torch.randn_like(t2_target).to(device)
                    gen_image = inferer.sample(input_noise=noise, diffusion_model=model, controlnet=cn_model, scheduler=inferer.scheduler, cn_cond=t1_cond, verbose=False)

                    images = torch.stack([
                        t1_cond.squeeze(0),
                        t2_target.squeeze(0),
                        normalize_image(gen_image.squeeze(0))
                    ], dim=0)

                    grid = torchvision.utils.make_grid(images, nrow=3)

                if batch_val_loss < best_val_loss:
                    run.log({
                        "best_model_generations": [wandb.Image(grid, caption=f"Epoch {e+1}")]
                    })

                    path = f'{CHECKPOINTS_PATH}/checkpoint_{exp_name}_{e+1}_best.pth'
                    torch.save({
                        'epoch': e + 1,
                        'model_state_dict': cn_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, path)
                    if best_model_path is not None and os.path.exists(best_model_path):
                        os.remove(best_model_path)
                    best_model_path = path
                    best_val_loss = batch_val_loss
                else:
                    run.log({
                        "each5e_generation": [wandb.Image(grid, caption=f"Epoch {e+1}")]
                    })
                    path = f'{CHECKPOINTS_PATH}/backups/checkpoint_{exp_name}_{e+1}.pth'
                    if os.path.exists(path):
                        os.remove(path)
                    torch.save({
                        'epoch': e + 1,
                        'model_state_dict': cn_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, path)
        end_time = time.time()
        elapsed_time = end_time - start_time
        run.log({"total_running_hours": elapsed_time // 3600})
        print(f"Training completed in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s")
    print("Training complete.")
    return best_model_path


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

    device = torch.device("cuda")
    num_train_timesteps = 1000

    df_model_path = 'test_path'
    if not os.path.exists(df_model_path):
        raise FileNotFoundError(f"Diffusion model checkpoint not found at {df_model_path}")
    df_checkpoint = torch.load(df_model_path, map_location=device)
    df_model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1
    )
    df_model.load_state_dict(df_checkpoint['model_state_dict'])
    df_model = df_model.to(device)

    controlnet = ControlNet(
        spatial_dims=2,
        in_channels=1,
        conditioning_embedding_num_channels=(32, )
    )

    controlnet.load_state_dict(df_model.state_dict(), strict=False)
    controlnet = controlnet.to(device)

    for param in df_model.parameters():
        param.requires_grad = False

    scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
    controlnet_inferer = ControlNetDiffusionInferer(scheduler)

    # ---------- Model training ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_name = f"controlnet_t1t2_{args.epochs}e"
    prediction_dir = f'{OUTPUT_DIR}/{exp_name}'
    num_train_timesteps = 1000
    project_name = 'Medical-I2I-Benchmark'

    best_model_path = train_controlnet(
        cn_model=controlnet,
        model=df_model,
        device=device,
        inferer=controlnet_inferer,
        train_loader=train_loader,
        val_loader=val_loader,
        project=project_name,
        exp_name=exp_name,
        notes="Training a ControlNet for T1-T2 brain image generation.",
        n_epochs=300,
        lr=1e-4,
        generation_steps=num_train_timesteps
    )

    # Load the best checkpoint
    controlnet.load_state_dict(torch.load(best_model_path, map_location=device)['model_state_dict'])
    controlnet.eval()
    # ---------- Model evaluation and prediction generation ----------
    with wandb.init(
        project=project_name,
        name=f'evaluation-{exp_name}',
        notes="Evaluation of the flow model on the test set.",
    ) as run:
        generate = make_generate_fn(df_model=df_model, inferer=controlnet_inferer)
        generate_and_save_predictions(controlnet, test_loader, device, output_dir=prediction_dir, generation_f=generate, wandb_run=run)
        out_dataset = PredictionDataset(directory=prediction_dir)
        summary = compute_ssim_from_dataset(out_dataset, wandb_run=run)
        print(summary)


if __name__ == "__main__":
    main()
