from torch.nn import CrossEntropyLoss
from torch.optim import *
from Net import *
from MDL_Net.MDL_Net import generate_model
from RLAD_Net.taad import get_model
from loss_function import joint_loss, loss_in_IMF
from utils.basic import get_scheduler
from HyperFusionNet.HyperFusion_AD_model import HyperFusion_AD
from vapformer.model_components import thenet
from thop import profile, clever_format
from Dataset import MriCliDataset
from torch.utils.data import DataLoader
# from torchsummary import summary
# compute the flops and params
num_classes = 2
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# Resnet
mri_demo = torch.ones(1, 1, 96, 128, 96).to(device)
pet_demo = torch.ones(1, 1, 96, 128, 96).to(device)
model_demo = ResnetMriPet(num_classes=num_classes).to(device)
model_demo.eval()
inputs = torch.cat([mri_demo, pet_demo], dim=1)
flops, params = profile(model_demo, inputs=(inputs,), verbose=False)
flops, params = clever_format([flops, params], "%.3f")
params_result = f'flops: {flops}, params: {params}'
print(f"Resnet:{params_result}")

# Vision Transformer
mri_demo = torch.ones(1, 1, 96, 128, 96).to(device)
pet_demo = torch.ones(1, 1, 96, 128, 96).to(device)
model_demo = ViTMriPet(num_classes=num_classes).to(device)
model_demo.eval()
inputs = torch.cat([mri_demo, pet_demo], dim=1)
flops, params = profile(model_demo, inputs=(inputs,), verbose=False)
flops, params = clever_format([flops, params], "%.3f")
params_result = f'flops: {flops}, params: {params}'
print(f"ViT:{params_result}")

# EfficientNet
mri_demo = torch.ones(1, 1, 96, 128, 96).to(device)
pet_demo = torch.ones(1, 1, 96, 128, 96).to(device)
model_demo = EfficientNetMriPet(num_classes=num_classes).to(device)
model_demo.eval()
inputs = torch.cat([mri_demo, pet_demo], dim=1)
flops, params = profile(model_demo, inputs=(inputs,), verbose=False)
flops, params = clever_format([flops, params], "%.3f")
params_result = f'flops: {flops}, params: {params}'
print(f"EfficientNet:{params_result}")

# Poolformer
mri_demo = torch.ones(1, 1, 96, 128, 96).to(device)
pet_demo = torch.ones(1, 1, 96, 128, 96).to(device)
model_demo = MetaFormerMriPet(num_classes=num_classes).to(device)
model_demo.eval()
inputs = torch.cat([mri_demo, pet_demo], dim=1)
flops, params = profile(model_demo, inputs=(inputs,), verbose=False)
flops, params = clever_format([flops, params], "%.3f")
params_result = f'flops: {flops}, params: {params}'
print(f"Poolformer:{params_result}")

# nnMamba
mri_demo = torch.ones(1, 1, 96, 128, 96).to(device)
pet_demo = torch.ones(1, 1, 96, 128, 96).to(device)
model_demo = nnMambaMriPet(num_classes=num_classes).to(device)
model_demo.eval()
inputs = torch.cat([mri_demo, pet_demo], dim=1)
flops, params = profile(model_demo, inputs=(inputs,), verbose=False)
flops,params = clever_format([flops, params], "%.3f")
params_result = f'flops: {flops}, params: {params}'
print(f"nnMamba:{params_result}")

#VAPL
mri_demo = torch.ones(1, 1, 113, 137, 113).to(device)
cli_demo = torch.ones(1, 9).to(device)
model_demo = thenet(input_size=[37 * 45 * 37, 18 * 22 * 18, 9 * 11 * 9, 4 * 5 * 4],dims=[32, 64, 128, 256],
                           depths=[3, 3, 3, 3], num_heads=8, in_channels=1,
                           num_classes=num_classes).to(device)
model_demo.eval()
flops, params = profile(model_demo, inputs=((mri_demo, cli_demo),), verbose=False)
flops, params = clever_format([flops, params], "%.3f")
params_result = f'flops: {flops}, params: {params}'
print(f"VAPL:{params_result}")

# IMF
mri_demo = torch.ones(1, 1, 96, 128, 96).to(device)
pet_demo = torch.ones(1, 1, 96, 128, 96).to(device)
cli_demo = torch.ones(1, 9).to(device)
model_demo = Interactive_Multimodal_Fusion_Model(num_classes=num_classes).to(device)
model_demo.eval()
flops, params = profile(model_demo, inputs=(mri_demo, pet_demo, cli_demo,), verbose=False)
flops, params = clever_format([flops, params], "%.3f")
params_result = f'flops: {flops}, params: {params}'
# summary_result = summary(model_demo, (3, 224, 224))
print(f"IMF:{params_result}")

# HFBSurv
mri_demo = torch.ones(2, 1, 96, 128, 96).to(device)
pet_demo = torch.ones(2, 1, 96, 128, 96).to(device)
cli_demo = torch.ones(2, 9).to(device)
model_demo = HFBSurv(num_classes=num_classes).to(device)
model_demo.eval()
result = model_demo(mri_demo, pet_demo, cli_demo)
print("result", result.shape)
flops, params = profile(model_demo, inputs=(mri_demo, pet_demo, cli_demo,), verbose=False)
flops, params = clever_format([flops, params], "%.3f")
params_result = f'flops: {flops}, params: {params}'
print(f"HFBSurv:{params_result}")

# ITCFN
mri_demo = torch.ones(1, 1, 96, 128, 96).to(device)
pet_demo = torch.ones(1, 1, 96, 128, 96).to(device)
cli_demo = torch.ones(1, 9).to(device)
model_demo = Triple_model_CoAttention_Fusion(num_classes=num_classes).to(device)
model_demo.eval()
flops, params = profile(model_demo, inputs=(mri_demo, pet_demo, cli_demo,), verbose=False)
flops, params = clever_format([flops, params], "%.3f")
params_result = f'flops: {flops}, params: {params}'
print(f"ITCFN:{params_result}")


# MDL
gm_demo = torch.ones(1, 1, 96, 128, 96).to(device)
wm_demo = torch.ones(1, 1, 96, 128, 96).to(device)
pet_demo = torch.ones(1, 1, 96, 128, 96).to(device)
inputs = torch.cat([gm_demo, wm_demo, pet_demo], dim=1)
# print("inputs", inputs.shape)
model_demo = generate_model(model_depth=18, in_planes=1, num_classes=num_classes).to(device)
model_demo.eval()
flops, params = profile(model_demo, inputs=(inputs, ), verbose=False)
flops, params = clever_format([flops, params], "%.3f")
params_result = f'flops: {flops}, params: {params}'
print(f"MDL:{params_result}")


# RLAD
mri_demo = torch.ones(1, 1, 128, 128, 128).to(device)
_, model_demo = get_model()
model_demo.to(device)
model_demo.eval()
flops, params = profile(model_demo, inputs=(mri_demo, ), verbose=False)
flops, params = clever_format([flops, params], "%.3f")
params_result = f'flops: {flops}, params: {params}'
print(f"RLAD:{params_result}")

# HyperFusionNet
mri_dir = '/data3/wangchangmiao/shenxy/ADNI/ADNI1_2/MRI'
pet_dir = '/data3/wangchangmiao/shenxy/ADNI/ADNI1_2/PET'
cli_dir = './csv/ADNI_Clinical.csv'
csv_file = './csv/ADNI1_2_pmci_smci.csv'
experiment_settings = {
    'shape': (96, 128, 96),
    'task': ('pMCI', 'sMCI')
}
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
dataset = MriCliDataset(mri_dir, pet_dir, cli_dir, csv_file,
                           resize_shape=experiment_settings['shape'],
                           valid_group=experiment_settings['task'])
dataloader = DataLoader(dataset, batch_size=505, shuffle=True, num_workers=0)
mri_demo = torch.ones(1, 1, 96, 128, 96)
cli_demo = torch.ones(1, 9)
model_demo = HyperFusion_AD(train_loader=dataloader, GPU=True, n_outputs=num_classes)
flops, params = profile(model_demo, inputs=((mri_demo, cli_demo),), verbose=False)
flops, params = clever_format([flops, params], "%.3f")
params_result = f'flops: {flops}, params: {params}'
print(f"HyperFusionNet:{params_result}")
"""
Resnet:flops: 70.924G, params: 66.952M
ViT:flops: 5.775G, params: 20.853M
EfficientNet:flops: 75.563M, params: 1.638M
Poolformer:flops: 9.549G, params: 31.210M
nnMamba:flops: 48.626G, params: 26.042M
VAPL:flops: 40.350G, params: 63.024M
IMF:flops: 70.925G, params: 67.843M
HFBSurv:flops: 141.849G, params: 34.123M
ITCFN:flops: 71.098G, params: 71.305M
HyperFusionNet:flops: 47.750G, params: 15.402M
MDL:flops: 9.353G, params: 2.827M
RLAD:flops: 260.882G, params: 30.624M
AweSomeNet:flops: 10.517G, params: 17.405M
"""