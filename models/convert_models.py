import torch

from generative.networks.nets import DiffusionModelUNet

best_model_to_convert_path="../your_model.pth"
onnx_path = "your_converted_model.onnx"
device = "cpu"

checkpoint = torch.load(best_model_to_convert_path, map_location="cpu")

# example for the UNet2PixT1T2 
unetG = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1
    )

unetG.load_state_dict(checkpoint['model_state_dict'])
unetG.eval()

batch_size = 1
channels = 1
height = 224  
width = 192

dummy_x = torch.randn(batch_size, channels, height, width, device=device)
dummy_t = torch.zeros(batch_size, device=device)

torch.onnx.export(
    unetG,
    (dummy_x, dummy_t),         
    onnx_path,
    export_params=True,
    opset_version=17,                 
    do_constant_folding=True,
    input_names=["x", "timesteps"],
    output_names=["out"],
    dynamic_axes={
        "x": {0: "batch"},
        "timesteps": {0: "batch"},
        "out": {0: "batch"}
    }
)

print(f"Model exported in: {onnx_path}")