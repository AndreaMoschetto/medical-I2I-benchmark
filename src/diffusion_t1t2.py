import os
import argparse

import torch
import time
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import Tensor
from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

import wandb
from tqdm import trange

from src.datasets import PredictionDataset, UnifiedBrainDataset
from src.utils import CHECKPOINTS_PATH, DATAPATH, OUTPUT_DIR, compute_ssim_from_dataset, ensure_checkpoint_dirs, generate_and_save_predictions, normalize_image

# -------------- Argument parser setup ----------

parser = argparse.ArgumentParser(description="Train a flow matching model from T1 to T2.")
parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate")
parser.add_argument('--lr_min', type=float, default=1e-6, help="Minimum learning rate for CosineAnnealingLR")
parser.add_argument('--num_workers', type=int, default=2, help="Number of workers for DataLoader")
parser.add_argument('--batchsize', type=int, default=6, help="Batch size")
parser.add_argument('--epochs', type=int, default=300, help="Number of training epochs")

args = parser.parse_args()

# -------------- Generation function ----------


def make_generate_fn(inferer: DiffusionInferer):
    @torch.no_grad()
    def generate_wrapped(model: nn.Module, condition: Tensor, gen_steps: int = 20):
        device = next(model.parameters()).device
        noise = torch.randn_like(condition).to(device)
        return inferer.sample(input_noise=noise, diffusion_model=model, scheduler=inferer.scheduler, mode='concat', conditioning=condition)
    return generate_wrapped
# ------------------ Training function ----------


def train_diffusion(model: DiffusionModelUNet, device: str, inferer: DiffusionInferer, train_loader: DataLoader, val_loader: DataLoader, project: str, exp_name: str, notes: str, n_epochs: int = 10, lr: float = 1e-3, generation_steps: int = 100):
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
        ensure_checkpoint_dirs()
        print("Using", torch.cuda.get_device_name(0)
              if torch.cuda.is_available() else "CPU")

        model.to(device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=args.lr_min)

        best_val_loss = float("inf")
        best_model_path = None
        start_time = time.time()
        for e in trange(n_epochs, desc="Epochs"):
            start_e_time = time.time()
            # Training
            model.train()
            train_losses = []
            for batch in train_loader:
                t2_targets = batch["t2"].to(device)  # [B, 1, H, W]
                t1_cond = batch["t1"].to(device)  # [B, 1, H, W]  # torch.randn_like(x_1).to(device)  # [B, 1, H, W]
                noise = torch.randn_like(t2_targets).to(device)

                B = t2_targets.shape[0]
                # Create timesteps
                timesteps = torch.randint(0, generation_steps, (B,), device=t2_targets.device).long()
                optimizer.zero_grad()
                # Get model prediction
                noise_pred = inferer(inputs=t2_targets, diffusion_model=model, noise=noise, timesteps=timesteps, condition=t1_cond, mode='concat')

                loss = criterion(noise_pred.float(), noise.float())
                train_losses.append(loss.item())

                loss.backward()
                optimizer.step()
            run.log({"train_loss": sum(train_losses) / len(train_losses)})
            run.log({"learning_rate": optimizer.param_groups[0]['lr']})

            # Validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    t2_targets = batch["t2"].to(device)
                    t1_cond = batch["t1"].to(device)
                    noise = torch.randn_like(t2_targets).to(device)
                    B = t2_targets.shape[0]
                    timesteps = torch.randint(0, generation_steps, (B,), device=t2_targets.device).long()
                    noise_pred = inferer(inputs=t2_targets, diffusion_model=model, noise=noise, timesteps=timesteps, condition=t1_cond, mode='concat')
                    val_loss = criterion(noise_pred.float(), noise.float())
                    val_losses.append(val_loss.item())
            batch_val_loss = sum(val_losses) / len(val_losses)
            run.log({"val_loss": batch_val_loss})
            e_time = time.time() - start_e_time
            run.log({"epoch_time_minutes": e_time // 60})

            lr_scheduler.step()
            # Checkpoint
            if e % 5 == 0 or e == n_epochs - 1 or batch_val_loss < best_val_loss:
                sample_batch = next(iter(val_loader))
                t1_gt = sample_batch["t1"][0].unsqueeze(0).to(device)
                t2_gt = sample_batch["t2"][0].to(device)
                noise = torch.randn_like(t1_gt).to(device)

                with torch.no_grad():
                    gen_image = inferer.sample(input_noise=noise, diffusion_model=model, scheduler=inferer.scheduler, mode='concat', conditioning=t1_gt)
                    images = torch.stack([t1_gt.squeeze(0), t2_gt, normalize_image(gen_image.squeeze(0))], dim=0)
                    grid = torchvision.utils.make_grid(images, nrow=3)

                if batch_val_loss < best_val_loss:
                    path = f'{CHECKPOINTS_PATH}/checkpoint_{exp_name}_{e+1}_best.pth'
                    torch.save({
                        'epoch': e + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, path)
                    if best_model_path is not None and os.path.exists(best_model_path):
                        os.remove(best_model_path)
                    best_model_path = path
                    best_val_loss = batch_val_loss
                    run.log({
                        "best_model_generations": [wandb.Image(grid, caption=f"Epoch {e+1} - Best")]
                    })
                else:
                    path = f'{CHECKPOINTS_PATH}/backups/checkpoint_{exp_name}_{e+1}.pth'
                    torch.save({
                        'epoch': e + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, path)
                    run.log({
                        "each5e_generation": [wandb.Image(grid, caption=f"Epoch {e+1}")]
                    })
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

    model = DiffusionModelUNet(
        spatial_dims=2,  # 2D
        in_channels=2,  # x
        out_channels=1,  # predicts noise
    )

    scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
    inferer = DiffusionInferer(scheduler)

    # ---------- Model training ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_name = f"unetflow-noiset1t2-s{args.epochs}e"
    prediction_dir = f'{OUTPUT_DIR}/{exp_name}'
    num_train_timesteps = 1000

    best_model_path = train_diffusion(
        model=model,
        inferer=inferer,
        train_loader=train_loader,
        val_loader=val_loader,
        project="FlowMatching-Baselines",
        exp_name=exp_name,
        notes="Training a diffusion model for T1-T2 brain image generation.",
        n_epochs=args.epochs,
        lr=args.lr,
        generation_steps=num_train_timesteps
    )

    # Load the best checkpoint
    model.load_state_dict(torch.load(best_model_path, map_location=device)['model_state_dict'])
    model.eval()
    # ---------- Model evaluation and prediction generation ----------
    with wandb.init(
        project='flowmatching-t1-to-t2',
        name=f'evaluation-{exp_name}',
        notes="Evaluation of the flow model on the test set.",
    ) as run:
        generate = make_generate_fn(inferer)
        generate_and_save_predictions(model, test_loader, device, output_dir=prediction_dir, generation_f=generate, wandb_run=run)
        out_dataset = PredictionDataset(directory=prediction_dir)
        summary = compute_ssim_from_dataset(out_dataset, wandb_run=run)

    summary


if __name__ == "__main__":
    main()
