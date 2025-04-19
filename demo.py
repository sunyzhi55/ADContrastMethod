from vapformer.model_components import thenet
import torch
model = thenet(
    input_size=[32 * 42 * 32, 16 * 21 * 16, 8 * 10 * 8, 4 * 5 * 4],
    dims=[32, 64, 128, 256],
    depths=[3, 3, 3, 3],
    num_heads=8,
    in_channels=1,
    num_classes=2
)
mri_image = torch.randn(1, 1, 96, 128, 96)
cli_tab = torch.randn(1, 9)
out = model((mri_image, cli_tab))
print(out.shape)