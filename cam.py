import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from Net.ComparisonNet import *
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
from skimage import transform as skt
import torchvision.transforms as transforms
import nibabel as nib


def get_clinical(sub_id, clin_df):
    '''Gets clinical features vector by searching dataframe for image id'''
    # 用-1初始化数组，表示缺失值
    clinical = np.full(9, -1.0)

    if sub_id in clin_df["PTID"].values:
        row = clin_df.loc[clin_df["PTID"] == sub_id].iloc[0]

        # GENDER (1表示Male, 2表示Female，缺失默认设置为 -1)
        if pd.isnull(row["PTGENDER"]):
            clinical[0] = -1
        else:
            clinical[0] = 1 if row["PTGENDER"] == 1 else (2 if row["PTGENDER"] == 2 else 0)

        # AGE (用-1标记缺失值)
        clinical[1] = row["AGE"] if not pd.isnull(row["AGE"]) else 0

        # Education (用-1标记缺失值)
        clinical[2] = row["PTEDUCAT"] if not pd.isnull(row["PTEDUCAT"]) else 0

        # FDG_bl (用-1标记缺失值)
        clinical[3] = row["FDG_bl"] if not pd.isnull(row["FDG_bl"]) else 0

        # TAU_bl (用-1标记缺失值)
        clinical[4] = row["TAU_bl"] if not pd.isnull(row["TAU_bl"]) else 0

        # PTAU_bl (用-1标记缺失值)
        clinical[5] = row["PTAU_bl"] if not pd.isnull(row["PTAU_bl"]) else 0

        # APOE4 (保留原有处理方式，缺失则处理为 -1)
        apoe4_allele = row["APOE4"]
        if pd.isnull(apoe4_allele):
            clinical[6], clinical[7], clinical[8] = 0, 0, 0  # 标记缺失值
        elif apoe4_allele == 0:
            clinical[6], clinical[7], clinical[8] = 1, 0, 0
        elif apoe4_allele == 1:
            clinical[6], clinical[7], clinical[8] = 0, 1, 0
        elif apoe4_allele == 2:
            clinical[6], clinical[7], clinical[8] = 0, 0, 1

    return clinical
class NoNan:  # Python3默认继承object类
    def __call__(self, data):  # __call___，让类实例变成一个可以被调用的对象，像函数
        nan_mask = np.isnan(data)
        data[nan_mask] = 0.0
        data = np.expand_dims(data, axis=0)
        data /= np.max(data)
        return data  # 返回预处理后的图像


class Numpy2Torch:  # Python3默认继承object类
    def __call__(self, data):  # __call___，让类实例变成一个可以被调用的对象，像函数
        data = torch.from_numpy(data)
        return data  # 返回预处理后的图像


class Resize:  # Python3默认继承object类
    def __init__(self, output_shape):
        self.output_shape = output_shape
    def __call__(self, data):  # __call___，让类实例变成一个可以被调用的对象，像函数
        data = skt.resize(data, output_shape=self.output_shape, order=1)
        return data  # 返回预处理后的图像
class MriPetCliDataset(Dataset):
    def __init__(self, mri_dir, pet_dir, cli_dir, csv_file, resize_shape=(96, 128, 96), valid_group=("pMCI", "sMCI")):
        """
        Args:
            mri_dir (string or Path): MRI 文件所在的文件夹路径。
            pet_dir (string or Path): PET 文件所在的文件夹路径。
            cli_dir (string or Path): Clinical 文件所在的文件夹路径。
            csv_file (string or Path): CSV 文件路径，其中第一列是文件名，第二列是标签。
            transform (callable, optional): 可选的转换操作，应用于样本。
        """
        self.mri_dir = Path(mri_dir)
        if pet_dir  == '':
            self.pet_dir = ''
        else:
            self.pet_dir = Path(pet_dir)
        self.cli_dir = pd.read_csv(cli_dir)
        self.labels_df = pd.read_csv(csv_file)  # 读取 CSV 文件
        self.groups = {'CN': 0, 'MCI': 1, 'AD': 2,  'pMCI': 1, 'sMCI': 0}
        self.valid_group = valid_group
        self.transform = transforms.Compose([
            Resize(resize_shape),
            NoNan(),
            Numpy2Torch(),
            # transforms.Normalize([0.5], [0.5])
        ])

        # 过滤只保留 valid_group 中的有效数据
        self.filtered_indices = self.labels_df[self.labels_df.iloc[:, 1].isin(self.valid_group)].index.tolist()

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        # 获取过滤后的索引
        filtered_idx = self.filtered_indices[idx]

        # 获取对应的文件名和标签
        img_name = self.labels_df.iloc[filtered_idx, 0]
        label_str = self.labels_df.iloc[filtered_idx, 1]  # 标签

        # MRI 文件路径
        mri_img_path = self.mri_dir / (img_name + '.nii')
        mri_img_numpy = nib.load(str(mri_img_path)).get_fdata()
        mri_img_torch = self.transform(mri_img_numpy)
        label = self.groups.get(label_str, -1)  # 获取标签，默认值为 -1

        clinical_features = get_clinical(img_name, self.cli_dir)
        clin_tab_torch = torch.from_numpy(clinical_features).float()

        # 只有MRI,没有PET,用于eval阶段
        # if self.pet_dir == '':
        #     return mri_img_torch.float(), label
        # PET 文件路径
        pet_img_path = self.pet_dir / (img_name + '.nii')
        # print('pet_img_path', pet_img_path)
        pet_img_numpy = nib.load(str(pet_img_path)).get_fdata()
        pet_img_torch = self.transform(pet_img_numpy)
        batch = {
            "mri": mri_img_torch.float(),
            "pet": pet_img_torch.float(),
            "clinical": clin_tab_torch,
            "label": label
        }

        return batch


def visualize_mri_pet_cam(model, inputs, target_category=1, slice_idx=48, use_cuda=True):
    """
    可视化MRI和PET的Grad-CAM热力图并叠加在原图上。

    Args:
        model: 已经加载好权重的EfficientNetMriPet模型
        inputs: MRI+PET输入, shape为 (batch, 2, 96, 128, 96)
        target_category: 解释哪一类的预测（默认1）
        slice_idx: 选择哪一张切片展示（默认48）
        use_cuda: 是否使用GPU（默认True）
    """
    device = torch.device('cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu')
    model = model.to(device).eval()
    inputs = inputs.to(device)
    print("inputs", inputs.shape)

    # 拆分MRI和PET
    mri = inputs[:, 0:1, :, :, :]  # (batch, 1, 96, 128, 96)
    pet = inputs[:, 1:2, :, :, :]

    # 分别给MRI和PET提特征
    mri_extractor = model.MriExtraction
    pet_extractor = model.PetExtraction

    # print("mri_extractor", mri_extractor)
    # 使用Grad-CAM
    cam_mri = GradCAM(model=mri_extractor, target_layers=[mri_extractor.layer2[-1]])
    cam_pet = GradCAM(model=pet_extractor, target_layers=[pet_extractor.layer2[-1]])
    # 定义目标类别
    targets = [ClassifierOutputTarget(target_category)]

    # 得到每个modal的CAM热力图 (batch, 1, D, H, W)，取出第一个样本
    cam_mri_img = cam_mri(input_tensor=mri, targets=targets)  # (D, H, W)
    cam_pet_img = cam_pet(input_tensor=pet, targets=targets)
    print("cam_mri_img", cam_mri_img.shape)
    print("cam_pet_img", cam_pet_img.shape)

    cam_mri_img = cam_mri_img[0]
    cam_pet_img = cam_pet_img[0]
    # 取出MRI原图和PET原图的对应slice
    mri_slice = mri[0, 0, :, :, slice_idx].detach().cpu().numpy()
    pet_slice = pet[0, 0, :, :, slice_idx].detach().cpu().numpy()

    # 归一化到 [0,1]
    def normalize_image(img):
        img_min, img_max = np.min(img), np.max(img)
        return (img - img_min) / (img_max - img_min + 1e-8)

    mri_slice_norm = normalize_image(mri_slice)
    pet_slice_norm = normalize_image(pet_slice)

    # 扩展成RGB
    mri_rgb = np.repeat(mri_slice_norm[:, :, np.newaxis], 3, axis=2)
    pet_rgb = np.repeat(pet_slice_norm[:, :, np.newaxis], 3, axis=2)

    # 叠加热力图到原图
    mri_cam_vis = show_cam_on_image(mri_rgb, cam_mri_img[:, :, slice_idx], use_rgb=True)
    pet_cam_vis = show_cam_on_image(pet_rgb, cam_pet_img[:, :, slice_idx], use_rgb=True)

    # 绘制结果
    print("result", mri_cam_vis)
    plt.figure(figsize=(10, 5))

    plt.subplot(2, 2, 1)
    plt.title(f'MRI Original (Slice {slice_idx})')
    plt.imshow(mri_slice_norm, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title(f'MRI with Grad-CAM')
    plt.imshow(mri_cam_vis)
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title(f'PET Original (Slice {slice_idx})')
    plt.imshow(pet_slice_norm, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title(f'PET with Grad-CAM')
    plt.imshow(pet_cam_vis)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('./result.png', dpi=300)
    plt.show()
if __name__ == '__main__':
    # model = EfficientNetMriPet(num_classes=2)
    model = ResnetMriPet(num_classes=2)

    # 记得要加载好你的权重（如果有的话）
    model.load_state_dict(torch.load('ResnetMriPet_best_model_fold2.pth', map_location="cuda:0"))
    print("model", model)
    model.eval()
    resize_shape = (96, 128, 96)
    self_transform = transforms.Compose([
            Resize(resize_shape),
            NoNan(),
            Numpy2Torch(),
            # transforms.Normalize([0.5], [0.5])
        ])
    mri_img_path = r'D:\dataset\final\freesurfer\ADNI2\MRI\003_S_4142.nii'
    mri_img_numpy = nib.load(str(mri_img_path)).get_fdata()
    mri_img_torch = self_transform(mri_img_numpy).float()
    mri_img_torch = mri_img_torch.unsqueeze(0)
    pet_img_path = r'D:\dataset\final\freesurfer\ADNI2\PET\003_S_4142.nii'
    pet_img_numpy = nib.load(str(pet_img_path)).get_fdata()
    pet_img_torch = self_transform(pet_img_numpy).float()
    pet_img_torch = pet_img_torch.unsqueeze(0)

    inputs = torch.concat([mri_img_torch, pet_img_torch], dim=1)

    # print("inputs", inputs.shape)
    visualize_mri_pet_cam(model, inputs, target_category=1, slice_idx=48)


