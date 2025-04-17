import nibabel as nib
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
from skimage import transform as skt
import numpy as np
import monai
from monai.data import DataLoader
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst,
    ScaleIntensity, RandFlip, RandAffine,
    RandGaussianNoise, RandGaussianSmooth,
    RandGibbsNoise, EnsureType, ToTensor,
    NormalizeIntensity
)


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


# class NoNan:  # Python3默认继承object类
#     def __call__(self, data):  # __call___，让类实例变成一个可以被调用的对象，像函数
#         nan_mask = np.isnan(data)
#         data[nan_mask] = 0.0
#         data = np.expand_dims(data, axis=0)
#         data /= np.max(data)
#         return data  # 返回预处理后的图像
#
#
# class Numpy2Torch:  # Python3默认继承object类
#     def __call__(self, data):  # __call___，让类实例变成一个可以被调用的对象，像函数
#         data = torch.from_numpy(data)
#         return data  # 返回预处理后的图像
#
#
# class Resize:  # Python3默认继承object类
#     def __init__(self, output_shape):
#         self.output_shape = output_shape
#     def __call__(self, data):  # __call___，让类实例变成一个可以被调用的对象，像函数
#         data = skt.resize(data, output_shape=self.output_shape, order=1)
#         return data  # 返回预处理后的图像

# MRI + PET + CLI + Two Label 数据集
class MriPetCliDatasetWithTowLabel(Dataset):
    def __init__(self, mri_dir, pet_dir, cli_dir, csv_file, resize_shape=(96, 128, 96), valid_group=("pMCI", "sMCI")):
        """
        Args:
            mri_dir (string or Path): MRI 文件所在的文件夹路径
            pet_dir (string or Path): PET 文件所在的文件夹路径
            cli_dir (string or Path): Clinical 文件所在的文件夹路径
            csv_file (string or Path): CSV 文件路径，其中第一列是文件名，第二列是标签
            resize_shape (tuple): 调整后的图像尺寸
            valid_group (tuple): 有效的标签组
        """
        self.mri_dir = Path(mri_dir)
        self.pet_dir = Path(pet_dir) if pet_dir else None
        self.cli_dir = pd.read_csv(cli_dir)
        self.labels_df = pd.read_csv(csv_file)
        self.groups = {'CN': 0, 'MCI': 1, 'AD': 2, 'pMCI': 1, 'sMCI': 0}
        self.valid_group = valid_group
        self.filtered_indices = self.labels_df[self.labels_df.iloc[:, 1].isin(self.valid_group)].index.tolist()
        self.convert_label = lambda x: [1, 0] if x == 0 else [0, 1]

        # monai 变换
        self.load_transform = Compose([
            LoadImage(image_only=True),  # 第一步：加载图像数据
            EnsureType()
        ])

        self.process_transform = Compose([
            EnsureChannelFirst(),  # 添加通道维度
            NormalizeIntensity(nonzero=True),
            ScaleIntensity(minv=0.0, maxv=1.0),  # 强度归一化
            monai.transforms.Resize(resize_shape),  # 调整尺寸
            EnsureType()  # 最终确保输出为tensor
        ])

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        filtered_idx = self.filtered_indices[idx]
        img_name = self.labels_df.iloc[filtered_idx, 0]
        label_str = self.labels_df.iloc[filtered_idx, 1]
        # 1. 加载MRI并处理NaN
        mri_path = str(self.mri_dir / (img_name + '.nii'))
        mri_data = self.load_transform(mri_path)
        mri_data = torch.nan_to_num(mri_data, nan=0.0)  # 处理NaN
        mri_img_torch = self.process_transform(mri_data)  # 应用后续转换

        # 加载PET
        pet_path = str(self.pet_dir / (img_name + '.nii'))
        pet_data = self.load_transform(pet_path)
        pet_data = torch.nan_to_num(pet_data, nan=0.0)
        pet_img_torch = self.process_transform(pet_data)

        # 临床特征
        clinical_features = get_clinical(img_name, self.cli_dir)
        clin_tab_torch = torch.from_numpy(clinical_features).float()

        # 标签处理
        label = self.groups.get(label_str, -1)
        label_2d = self.convert_label(label)

        return {
            "mri": mri_img_torch,
            "pet": pet_img_torch,
            "clinical": clin_tab_torch,
            "label": label,
            "label_2d": torch.Tensor(label_2d)
        }


# MRI + PET + CLI 数据集
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
        # 过滤只保留 valid_group 中的有效数据
        self.mri_dir = Path(mri_dir)
        self.pet_dir = Path(pet_dir) if pet_dir else ''
        self.cli_dir = pd.read_csv(cli_dir)
        self.labels_df = pd.read_csv(csv_file)
        self.groups = {'CN': 0, 'MCI': 1, 'AD': 2, 'pMCI': 1, 'sMCI': 0}
        self.valid_group = valid_group
        self.filtered_indices = self.labels_df[self.labels_df.iloc[:, 1].isin(self.valid_group)].index.tolist()
        # monai 变换
        self.load_transform = Compose([
            LoadImage(image_only=True),  # 第一步：加载图像数据
            EnsureType()
        ])
        self.process_transform = Compose([
            EnsureChannelFirst(),  # 添加通道维度
            NormalizeIntensity(nonzero=True),
            ScaleIntensity(minv=0.0, maxv=1.0),  # 强度归一化
            monai.transforms.Resize(resize_shape),  # 调整尺寸
            EnsureType()  # 最终确保输出为tensor
        ])

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        # 获取过滤后的索引
        filtered_idx = self.filtered_indices[idx]

        # 获取对应的文件名和标签
        img_name = self.labels_df.iloc[filtered_idx, 0]
        label_str = self.labels_df.iloc[filtered_idx, 1]  # 标签

        # 1. 加载MRI并处理NaN
        mri_path = str(self.mri_dir / (img_name + '.nii'))
        mri_data = self.load_transform(mri_path)
        mri_data = torch.nan_to_num(mri_data, nan=0.0)  # 处理NaN
        mri_img_torch = self.process_transform(mri_data)  # 应用后续转换

        # 加载PET
        # 只有MRI,没有PET,用于eval阶段
        # if self.pet_dir == '':
        #     return mri_img_torch.float(), label
        pet_path = str(self.pet_dir / (img_name + '.nii'))
        pet_data = self.load_transform(pet_path)
        pet_data = torch.nan_to_num(pet_data, nan=0.0)
        pet_img_torch = self.process_transform(pet_data)

        label = self.groups.get(label_str, -1)  # 获取标签，默认值为 -1
        clinical_features = get_clinical(img_name, self.cli_dir)
        clin_tab_torch = torch.from_numpy(clinical_features).float()
        batch = {
            "mri": mri_img_torch.float(),
            "pet": pet_img_torch.float(),
            "clinical": clin_tab_torch,
            "label": label
        }
        return batch


# MRI + PET 数据集
class MriPetDataset(Dataset):
    def __init__(self, mri_dir, pet_dir, cli_dir, csv_file, resize_shape=(96, 128, 96), valid_group=("pMCI", "sMCI")):
        """
        Args:
            mri_dir (string or Path): MRI 文件所在的文件夹路径。
            pet_dir (string or Path): PET 文件所在的文件夹路径。
            csv_file (string or Path): CSV 文件路径，其中第一列是文件名，第二列是标签。
            transform (callable, optional): 可选的转换操作，应用于样本。
        """
        self.mri_dir = Path(mri_dir)
        if pet_dir == '':
            self.pet_dir = ''
        else:
            self.pet_dir = Path(pet_dir)
        self.labels_df = pd.read_csv(csv_file)  # 读取 CSV 文件
        self.groups = {'CN': 0, 'MCI': 1, 'AD': 2, 'pMCI': 1, 'sMCI': 0}
        self.valid_group = valid_group
        # monai 变换
        self.load_transform = Compose([
            LoadImage(image_only=True),  # 第一步：加载图像数据
            EnsureType()
        ])
        self.process_transform = Compose([
            EnsureChannelFirst(),  # 添加通道维度
            NormalizeIntensity(nonzero=True),
            ScaleIntensity(minv=0.0, maxv=1.0),  # 强度归一化
            monai.transforms.Resize(resize_shape),  # 调整尺寸
            EnsureType()  # 最终确保输出为tensor
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

        # 1. 加载MRI并处理NaN
        mri_path = str(self.mri_dir / (img_name + '.nii'))
        mri_data = self.load_transform(mri_path)
        mri_data = torch.nan_to_num(mri_data, nan=0.0)  # 处理NaN
        mri_img_torch = self.process_transform(mri_data)  # 应用后续转换
        # 加载PET
        # 只有MRI,没有PET,用于eval阶段
        # if self.pet_dir == '':
        #     return mri_img_torch.float(), label
        pet_path = str(self.pet_dir / (img_name + '.nii'))
        pet_data = self.load_transform(pet_path)
        pet_data = torch.nan_to_num(pet_data, nan=0.0)
        pet_img_torch = self.process_transform(pet_data)
        label = self.groups.get(label_str, -1)  # 获取标签，默认值为 -1
        batch = {
            "mri": mri_img_torch.float(),
            "pet": pet_img_torch.float(),
            "label": label
        }
        return batch


# 单MRI 数据集
class MriDataset(Dataset):
    def __init__(self, mri_dir, pet_dir, cli_dir, csv_file, resize_shape=(96, 128, 96), valid_group=("pMCI", "sMCI")):
        """
        Args:
            mri_dir (string or Path): MRI 文件所在的文件夹路径。
            pet_dir (string or Path): PET 文件所在的文件夹路径。
            csv_file (string or Path): CSV 文件路径，其中第一列是文件名，第二列是标签。
            transform (callable, optional): 可选的转换操作，应用于样本。
        """
        self.mri_dir = Path(mri_dir)
        if pet_dir == '':
            self.pet_dir = ''
        else:
            self.pet_dir = Path(pet_dir)
        self.labels_df = pd.read_csv(csv_file)  # 读取 CSV 文件
        self.groups = {'CN': 0, 'MCI': 1, 'AD': 2, 'pMCI': 1, 'sMCI': 0}
        self.valid_group = valid_group
        # monai 变换
        self.load_transform = Compose([
            LoadImage(image_only=True),  # 第一步：加载图像数据
            EnsureType()
        ])
        self.process_transform = Compose([
            EnsureChannelFirst(),  # 添加通道维度
            NormalizeIntensity(nonzero=True),
            ScaleIntensity(minv=0.0, maxv=1.0),  # 强度归一化
            monai.transforms.Resize(resize_shape),  # 调整尺寸
            EnsureType()  # 最终确保输出为tensor
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

        # 1. 加载MRI并处理NaN
        mri_path = str(self.mri_dir / (img_name + '.nii'))
        mri_data = self.load_transform(mri_path)
        mri_data = torch.nan_to_num(mri_data, nan=0.0)  # 处理NaN
        mri_img_torch = self.process_transform(mri_data)  # 应用后续转换
        label = self.groups.get(label_str, -1)  # 获取标签，默认值为 -1

        batch = {
            "mri": mri_img_torch.float(),
            "label": label
        }

        return batch


# MRI + CLI 数据集
class MriCliDataset(Dataset):
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
        self.cli_dir = pd.read_csv(cli_dir)
        self.labels_df = pd.read_csv(csv_file)  # 读取 CSV 文件
        self.groups = {'CN': 0, 'MCI': 1, 'AD': 2, 'pMCI': 1, 'sMCI': 0}
        self.valid_group = valid_group
        self.only_tabular = False
        # monai 变换
        self.load_transform = Compose([
            LoadImage(image_only=True),  # 第一步：加载图像数据
            EnsureType()
        ])
        self.process_transform = Compose([
            EnsureChannelFirst(),  # 添加通道维度
            NormalizeIntensity(nonzero=True),
            ScaleIntensity(minv=0.0, maxv=1.0),  # 强度归一化
            monai.transforms.Resize(resize_shape),  # 调整尺寸
            EnsureType()  # 最终确保输出为tensor
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

        # 1. 加载MRI并处理NaN
        mri_path = str(self.mri_dir / (img_name + '.nii'))
        mri_data = self.load_transform(mri_path)
        mri_data = torch.nan_to_num(mri_data, nan=0.0)  # 处理NaN
        mri_img_torch = self.process_transform(mri_data)  # 应用后续转换
        label = self.groups.get(label_str, -1)  # 获取标签，默认值为 -1
        clinical_features = get_clinical(img_name, self.cli_dir)
        clin_tab_torch = torch.from_numpy(clinical_features).float()
        batch = {
            "mri": mri_img_torch.float(),
            "clinical": clin_tab_torch,
            "label": label
        }
        return batch


# 自定义 Dataset 类 GM WM PET
class GMWMPETDataset(Dataset):
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
        self.GM_dir = Path(mri_dir) / 'GM'
        self.WM_dir = Path(mri_dir) / 'WM'
        self.pet_dir = Path(pet_dir)
        self.cli_dir = pd.read_csv(cli_dir)
        self.labels_df = pd.read_csv(csv_file)  # 读取 CSV 文件
        self.groups = {'CN': 0, 'MCI': 1, 'AD': 2, 'pMCI': 1, 'sMCI': 0}
        self.valid_group = valid_group
        # monai 变换
        self.load_transform = Compose([
            LoadImage(image_only=True),  # 第一步：加载图像数据
            EnsureType()
        ])
        self.process_transform = Compose([
            EnsureChannelFirst(),  # 添加通道维度
            NormalizeIntensity(nonzero=True),
            ScaleIntensity(minv=0.0, maxv=1.0),  # 强度归一化
            monai.transforms.Resize(resize_shape),  # 调整尺寸
            EnsureType()  # 最终确保输出为tensor
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


        # 1. 加载MRI并处理NaN
        gm_img_path = str(self.GM_dir / (img_name + '.nii'))
        gm_img_data = self.load_transform(gm_img_path)
        gm_img_data = torch.nan_to_num(gm_img_data, nan=0.0)  # 处理NaN
        gm_img_torch = self.process_transform(gm_img_data)  # 应用后续转换

        # WM 文件路径

        # 1. 加载MRI并处理NaN
        wm_img_path = str(self.WM_dir / (img_name + '.nii'))
        wm_img_data = self.load_transform(wm_img_path)
        wm_img_data = torch.nan_to_num(wm_img_data, nan=0.0)  # 处理NaN
        wm_img_torch = self.process_transform(wm_img_data)  # 应用后续转换

        # PET 文件路径
        pet_path = str(self.pet_dir / (img_name + '.nii'))
        pet_data = self.load_transform(pet_path)
        pet_data = torch.nan_to_num(pet_data, nan=0.0)
        pet_img_torch = self.process_transform(pet_data)
        label = self.groups.get(label_str, -1)  # 获取标签，默认值为 -1
        batch = {
            "gm": gm_img_torch.float(),
            "wm": wm_img_torch.float(),
            "pet": pet_img_torch.float(),
            "label": label
        }
        return batch



if __name__ == '__main__':
    mri_dir = r'/data3/wangchangmiao/shenxy/ADNI/ADNI2/MRI'  # 替换为 MRI 文件的路径
    pet_dir = r'/data3/wangchangmiao/shenxy/ADNI/ADNI2/PET'  # 替换为 PET 文件的路径
    cli_dir = r'./csv/ADNI_Clinical.csv'
    csv_file = r'./csv/ADNI2_match.csv'  # 替换为 CSV 文件路径
    batch_size = 8  # 设置批次大小

    dataset = MriPetCliDatasetWithTowLabel(mri_dir, pet_dir, cli_dir, csv_file, valid_group=("pMCI", "sMCI"))
    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # labels = [data["label"] for data in dataset]  # 假设dataset[i]的第3项是label
    # print("labels", labels)

    # 测试读取数据
    print('dataloader', len(dataloader))
    print('dataset', len(dataloader.dataset))
    for batch in dataloader:
        mri_batch = batch["mri"]  # shape: [B, 1, D, H, W]
        pet_batch = batch.get("pet")  # shape: [B, 1, D, H, W]（如果有）
        clin_batch = batch["clinical"]  # shape: [B, N_features]
        labels = batch["label"]  # shape: [B]

        # (mri_imgs, pet_imgs, cli_tab, labels) = batch
        if torch.isnan(mri_batch).any():
            print("train: NaN detected in input mri_images")
        if torch.isnan(pet_batch).any():
            print("train: NaN detected in input pet_images")
        print(f" MRI Images batch shape: {mri_batch.shape}, type:{type(mri_batch)}")  # ([8, 1, 96, 128, 96])
        print(f" PET Images batch shape: {pet_batch.shape}, type:{type(pet_batch)}")  # ([8, 1, 96, 128, 96])
        print(f" Clinical Table batch shape: {clin_batch.shape}, type:{type(clin_batch)}")  # ([8, 9])
        print(f" Labels batch shape: {labels}, type:{type(labels)} ")
