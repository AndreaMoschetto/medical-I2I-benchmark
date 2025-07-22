from skimage import filters
import os
import numpy as np
import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim_fn, peak_signal_noise_ratio as psnr_fn
import torch.nn.functional as F
from wandb.sdk.wandb_run import Run
from dotenv import load_dotenv

# Carica variabili da file .env (se presente)
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATAPATH = os.getenv("MEDICAL_I2I_DATAPATH", os.path.join(BASE_DIR, "data"))
OUTPUT_DIR = os.getenv("MEDICAL_I2I_OUTPUT", os.path.join(BASE_DIR, "outputs"))
CHECKPOINTS_PATH = os.getenv("MEDICAL_I2I_CKPT", os.path.join(BASE_DIR, "checkpoints"))
BACKUP_PATH = os.path.join(CHECKPOINTS_PATH, "backups")

for path in [DATAPATH, OUTPUT_DIR, CHECKPOINTS_PATH, BACKUP_PATH]:
    os.makedirs(path, exist_ok=True)


def ensure_checkpoint_dirs():
    if not os.path.exists(BACKUP_PATH):
        os.makedirs(BACKUP_PATH)


def percnorm(arr, lperc=5, uperc=99.5):
    """
    Remove outlier intensities from a brain component,
    similar to Tukey's fences method.
    """
    upperbound = np.percentile(arr, uperc)
    lowerbound = np.percentile(arr, lperc)
    arr[arr > upperbound] = upperbound
    arr[arr < lowerbound] = lowerbound
    return arr


def normalize(img):
    # img: [C, H, W]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


def normalize_image(in_tensor_img):
    # tensor_img: [1, H, W]
    img_np = in_tensor_img.squeeze(0).cpu().numpy()  # [H, W]
    # Percentile-based normalization
    img_np = percnorm(img_np)
    out_tensor_image = torch.from_numpy(
        img_np).unsqueeze(0)  # Back to [1, H, W]
    out_tensor_image = normalize(
        out_tensor_image)            # 0-1 normalization
    return out_tensor_image.to(in_tensor_img.device)


@torch.no_grad()
def generate_and_save_predictions(model, test_loader, device, output_dir=OUTPUT_DIR, generation_f: callable = None, wandb_run: Run = None, just_one_batch: bool = False):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    all_outputs = []

    for idx, batch in enumerate(tqdm(test_loader, desc="Generating Predictions")):
        t1 = batch["t1"].to(device)           # [B, 1, H, W]
        t2 = batch["t2"].to(device)           # [B, 1, H, W]
        filenames = batch["filename"]         # list of strings (length B)

        x_gen = generation_f(model, condition=t1, gen_steps=300)

        for i in range(t1.size(0)):
            sample = {
                "filename": filenames[i],
                "input": t1[i].cpu(),         # torch.Tensor [1, H, W]
                "target": t2[i].cpu(),
                "prediction": x_gen[i].cpu()
            }

            torch.save(sample, os.path.join(output_dir, f"{filenames[i]}.pt"))
            all_outputs.append(sample)
        if just_one_batch:
            break
        if wandb_run:
            wandb_run.log({"prediction_progress": idx})

    return all_outputs


def compute_ssim_from_dataset(dataset, wandb_run: Run = None):
    ssim_scores = []
    mse_scores = []
    ssim_masked_scores = []
    psnr_scores = []

    example_pred = None
    example_gt = None

    for i in range(len(dataset)):
        pred, gt = dataset[i]  # tensors [1, H, W]

        # Convert to numpy and squeeze channel
        pred_np = pred.squeeze().cpu().numpy()
        gt_np = gt.squeeze().cpu().numpy()

        pred_np = normalize(percnorm(pred_np))
        gt_np = normalize(percnorm(gt_np))

        threshold_triangle_gt = filters.threshold_triangle(gt_np)
        mask = gt_np > threshold_triangle_gt
        masked_gt = gt_np * mask

        threshold_triangle_pred = filters.threshold_triangle(pred_np)
        mask_pred = pred_np > threshold_triangle_pred
        masked_pred = pred_np * mask_pred

        # Compute SSIM
        ssim_masked = ssim_fn(masked_gt, masked_pred, data_range=1.0)
        ssim_masked_scores.append(ssim_masked)

        ssim_val = ssim_fn(pred_np, gt_np, data_range=1.0)
        ssim_scores.append(ssim_val)

        # Compute MSE
        mse_val = F.mse_loss(pred, gt).item()
        mse_scores.append(mse_val)

        # Compute PSNR on masked images
        psnr_val = psnr_fn(masked_gt, masked_pred, data_range=1.0)
        psnr_scores.append(psnr_val)

        # Store one example for visualization
        if i == 4 and example_pred is None:
            example_pred = masked_pred
            example_gt = masked_gt

    ssim_scores = np.array(ssim_scores)
    ssim_masked_scores = np.array(ssim_masked_scores)
    mse_scores = np.array(mse_scores)
    psnr_scores = np.array(psnr_scores)

    summary = pd.DataFrame({
        "Metric": ["SSIM", "MASKED_SSIM", "MSE", "PSNR"],
        "Mean": [ssim_scores.mean(), ssim_masked_scores.mean(), mse_scores.mean(), psnr_scores.mean()],
        "Variance": [ssim_scores.var(), ssim_masked_scores.var(), mse_scores.var(), psnr_scores.var()],
    })

    # Visualize example
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(example_gt, cmap='gray')
    axs[0].set_title("Ground Truth")
    axs[0].axis("off")

    axs[1].imshow(example_pred, cmap='gray')
    axs[1].set_title("Prediction")
    axs[1].axis("off")

    plt.suptitle("Example Comparison")
    plt.tight_layout()
    plt.show()

    if wandb_run:
        wandb_run.log({"eval/ssim_mean": summary["Mean"][0]})
        wandb_run.log({"eval/ssim_masked_mean": summary["Mean"][1]})
        wandb_run.log({"eval/mse_mean": summary["Mean"][2]})
        wandb_run.log({"eval/psnr_mean": summary["Mean"][3]})
        wandb_run.log({"eval/ssim_var": summary["Variance"][0]})
        wandb_run.log({"eval/ssim_masked_var": summary["Variance"][1]})
        wandb_run.log({"eval/mse_var": summary["Variance"][2]})
        wandb_run.log({"eval/psnr_var": summary["Variance"][3]})

    return summary


# Initialize weights (optional, often helpful for GANs)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
