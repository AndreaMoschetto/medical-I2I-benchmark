import os
import argparse

import torch
import time
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import Tensor
from generative.networks.nets import DiffusionModelUNet
import wandb
from tqdm import tqdm, trange

from t1t2converter.datasets import PredictionDataset, UnifiedBrainDataset
from t1t2converter.utils import CHECKPOINTS_PATH, DATAPATH, OUTPUT_DIR, compute_ssim_from_dataset, ensure_checkpoint_dirs, generate_and_save_predictions, normalize_image

# -------------- Argument parser setup ----------

parser = argparse.ArgumentParser(description="Train a flow matching model from T1 to T2.")
parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate")
parser.add_argument('--lr_min', type=float, default=1e-6, help="Minimum learning rate for CosineAnnealingLR")
parser.add_argument('--num_workers', type=int, default=2, help="Number of workers for DataLoader")
parser.add_argument('--batchsize', type=int, default=6, help="Batch size")
parser.add_argument('--epochs', type=int, default=300, help="Number of training epochs")

args = parser.parse_args()

# -------------- Generation function ----------


def euler_step(model: DiffusionModelUNet, x_t: Tensor, t_start: Tensor, t_end: Tensor):
    # delta_t shape (B, 1, 1, 1)
    delta_t = (t_end - t_start).view(-1, 1, 1, 1)

    # model si aspetta t come tensor (B,)
    v_hat = model(x_t, t_start)

    x_t_noise = x_t[:, 0:1, :, :]  # [B, 1, H, W]
    x_t_cond = x_t[:, 1:2, :, :]  # [B, 1, H, W], che è T1

    x_next_noise = x_t_noise + delta_t * v_hat

    x_next = torch.cat([x_next_noise, x_t_cond], dim=1)  # [B, 2, H, W]
    return x_next


@torch.no_grad()
def generate(model: nn.Module, condition: Tensor, gen_steps: int = 20):
    model.eval()

    device = condition.device
    batch_size = condition.shape[0]

    time_steps = torch.linspace(0.0, 1.0, gen_steps + 1, device=device, dtype=torch.float32)

    x = torch.cat([torch.randn_like(condition, device=device), condition], dim=1)  # [B, 2, H, W]
    for i in range(gen_steps):
        t_start = time_steps[i].expand(batch_size)
        t_end = time_steps[i + 1].expand(batch_size)
        x = euler_step(model, x_t=x, t_start=t_start, t_end=t_end)

    return x[:, 0:1, :, :]  # [B, 1, H, W]


# ------------------ Training function ----------

def train_flow(model: DiffusionModelUNet, device: str, train_loader: DataLoader, val_loader: DataLoader, project: str, exp_name: str, notes: str, n_epochs: int = 10, lr: float = 1e-3, generation_steps: int = 100):
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
            'learning_rate': f'{lr} -> 1e-6',
            'loss_function': 'MSELoss',
            'generation_steps': generation_steps,
            'device': str(torch.cuda.get_device_name(0)
                          if torch.cuda.is_available() else "CPU"),
        }
    ) as run:
        ensure_checkpoint_dirs()
        print("Using", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

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
            for batch in tqdm(train_loader, desc=f"Training epoch {e}"):
                x_1 = batch["t2"].to(device)  # [B, 1, H, W]
                x_0_cond = batch["t1"].to(device)  # [B, 1, H, W]  # torch.randn_like(x_1).to(device)  # [B, 1, H, W]
                x_0_noise = torch.randn_like(x_0_cond).to(device)  # [B, 1, H, W]

                # add the corresponding t1 to the second channel of x_0
                B = x_0_cond.shape[0]
                t = torch.rand(B, device=device)  # B
                t_img = t.view(B, 1, 1, 1)  # [B, 1, 1, 1] for broadcasting

                x_t = (1 - t_img) * x_0_noise + t_img * x_1  # [B, 1, H, W]
                x_t = torch.cat([x_t, x_0_cond], dim=1)  # [B, 2, H, W]

                dx_t = x_1 - x_0_noise  # [B, 1, H, W]

                optimizer.zero_grad()
                pred = model(x_t, t)  # [B, 1, H, W]
                loss = criterion(pred, dx_t)
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
                    x_1 = batch["t2"].to(device)
                    x_0_cond = batch["t1"].to(device)
                    x_0_noise = torch.randn_like(x_0_cond).to(device)  # [B, 1, H, W]
                    B = x_0_cond.shape[0]
                    t = torch.rand(B, device=device)
                    t_img = t.view(B, 1, 1, 1)
                    x_t = (1 - t_img) * x_0_noise + t_img * x_1
                    x_t = torch.cat([x_t, x_0_cond], dim=1)  # [B, 2, H, W]
                    dx_t = x_1 - x_0_noise

                    pred = model(x_t, t)
                    val_loss = criterion(pred, dx_t)
                    val_losses.append(val_loss.item())
            batch_val_loss = sum(val_losses) / len(val_losses)
            run.log({"val_loss": batch_val_loss})
            e_time = time.time() - start_e_time
            run.log({"epoch_time_minutes": e_time // 60})

            lr_scheduler.step()
            # Checkpoint
            if e % 5 == 0 or e == n_epochs - 1 or batch_val_loss < best_val_loss:
                sample_batch = next(iter(val_loader))  # just one batch
                t1_gt = sample_batch["t1"][0].unsqueeze(0).to(device)
                t2_gt = sample_batch["t2"][0].to(device)  # [1, 1, H, W]
                t2_gen = generate(model=model, condition=t1_gt, gen_steps=generation_steps)  # [1, 1, H, W]
                imgs = torch.stack([t1_gt.squeeze(0), t2_gt, normalize_image(t2_gen.squeeze(0))], dim=0)
                grid = torchvision.utils.make_grid(imgs, nrow=3)

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

    model = DiffusionModelUNet(
        spatial_dims=2,  # 2D
        in_channels=2,  # x + noise
        out_channels=1,  # predicts delta_x_t
    )
    # ---------- Model training ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_name = f"unetflow-noiset1t2-s{args.epochs}e"
    prediction_dir = f'{OUTPUT_DIR}/{exp_name}'
    project_name = 'flowmatching-t1-to-t2'

    best_modelpath = train_flow(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        project=project_name,
        exp_name=exp_name,
        notes="Small UNet flow model for directional diffusion from T1 + noise to T2.",
        n_epochs=args.epochs,
        lr=args.lr,
        generation_steps=300)

    # Load the best checkpoint
    model.load_state_dict(torch.load(best_modelpath, map_location=device)['model_state_dict'])
    model.eval()
    # ---------- Model evaluation and prediction generation ----------
    with wandb.init(
        project=project_name,
        name=f'evaluation-{exp_name}',
        notes="Evaluation of the flow model on the test set.",
    ) as run:
        generate_and_save_predictions(model, test_loader, device, output_dir=prediction_dir, generation_f=generate, wandb_run=run)
        out_dataset = PredictionDataset(directory=prediction_dir)
        summary = compute_ssim_from_dataset(out_dataset, wandb_run=run)

        print(summary)


if __name__ == "__main__":
    main()
