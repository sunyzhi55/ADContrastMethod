from HyperFusionNet.base_models import *
from HyperFusionNet.hyper_base import *

import nibabel as nib
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
from skimage import transform as skt
import numpy as np
import torch
from torch import nn
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

class HyperFusion_AD(nn.Module):
    # in_channels=1, n_outputs=2, bn_momentum=0.05, init_features=16, n_tabular_features=9,
    def __init__(self, in_channels=1, n_outputs=2, bn_momentum=0.05, init_features=32, n_tabular_features=9, **kwargs):
        super().__init__()

        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None

        embd_tab_out_size = 8
        # embd_model_path_end = f'baseline-tabular_v1-seed{kwargs["split_seed"]}-fs{kwargs["features_set"]}/fold_{kwargs["data_fold"]}/best_val.ckpt'
        # embd_model_path = os.path.join(kwargs["checkpoint_dir"], embd_model_path_end)
        # model = PlModelWrapADcls.load_from_checkpoint(embd_model_path).model.mlp
        # hyper_embeddings_tab = nn.Sequential(
        #     model[0],
        #     model[1],
        #     model[2]
        # )
        hyper_embeddings_tab = nn.Sequential(
            nn.Linear(n_tabular_features, 32),
            nn.ReLU(),
            nn.Linear(32, embd_tab_out_size)
        )# .to("cuda:2")

        #                                               [conv1 hyper, conv2 hyper, down-sample hyper]
        resblock_hyper_kwargs = dict(hyper_embedding_models=[None, None, hyper_embeddings_tab])
        fc1_hyper_kwargs = dict(embedding_model=None)
        fc2_hyper_kwargs = dict(embedding_model=None)


        general_hyper_kwargs = dict(
            embedding_output_size=embd_tab_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="tabular",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
        )
        resblock_hyper_kwargs.update(general_hyper_kwargs)
        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block1 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block3 = PreactivResBlock_bn(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block4 = HyperPreactivResBlock(8 * init_features, 16 * init_features, bn_momentum=bn_momentum,
                                            stride=2, dropout=0.3, **resblock_hyper_kwargs)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(16*init_features, 4*init_features, **fc1_hyper_kwargs)

        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(4*init_features, n_outputs, **fc2_hyper_kwargs)

        self.relu = nn.ReLU()

    def forward(self, x):
        image, tabular = x

        out = self.conv_bn_relu(image)
        out = self.max_pool3d_1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4((out, tabular))
        out = self.adaptive_avg_pool3d(out)
        out = out.view(out.size(0), -1)
        out = self.linear_drop1(out)
        out = self.fc1((out, tabular))
        out = self.relu(out)
        out = self.linear_drop2(out)
        out = self.fc2((out, tabular))

        return out


# 测试模型
# if __name__ == "__main__":
#     B = 2
#     n_tabular_features = 9
#     model = HyperFusion_AD(
#         in_channels=1,
#         n_outputs=3,
#         init_features=4,
#         n_tabular_features=n_tabular_features,
#     )
#
#     image_tensor = torch.randn(B, 1, 96, 128, 96)
#     tabular_tensor = torch.randn(B, n_tabular_features)
#
#     out = model((image_tensor, tabular_tensor))
#     print("✅ Output shape:", out.shape)  # 应该是 [2, 3]
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
        self.groups = {'DM': 1, 'AD': 1, 'CN': 0, 'pMCI': 1, 'sMCI': 0, 'sSCD': 0, 'pSCD': 1,
                       'MCI': 1, 'sSMC': 0, 'pSMC': 1, 'SMC': 0, 'sCN': 0,
                       'pCN': 1, 'ppCN': 1, 'Autism': 1, 'Control': 0}
        self.valid_group = valid_group
        self.only_tabular = False
        self.transform = transforms.Compose([
            Resize(resize_shape),
            NoNan(),
            Numpy2Torch(),
            transforms.Normalize([0.5], [0.5])
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
        return mri_img_torch.float(), clin_tab_torch, label
#
# if __name__ == "__main__":
#     # 假设 tabular 有 9 维特征（比如年龄、性别、量表等）
#     n_tabular_features = 9
#     dataset = MriCliDataset(
#         mri_dir=r'D:\dataset\final\freesurfer\ADNI1\mini\MRI',
#         pet_dir=r'D:\dataset\final\freesurfer\ADNI1\mini\MRI',
#         cli_dir=r'D:\dataset\final\freesurfer\ADNI1\mini\ADNI_Clinical.csv',
#         csv_file=r'D:\dataset\final\freesurfer\ADNI1\mini\ADNI1_all.csv',
#         resize_shape=(96, 128, 96),
#         valid_group=("pMCI", "sMCI")
#     )
#     train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("device", device)
#     model = HyperFusion_AD(train_loader=train_loader, GPU=True,).to(device)
#
#     # 随机生成一个 MRI + 表格组合输入
#     image_tensor = torch.randn(2, 1, 96, 128, 96).to(device)  # [batch_size, channels, D, H, W]
#     tabular_tensor = torch.randn(2, n_tabular_features).to(device)  # [batch_size, tabular_features]
#     from thop import profile, clever_format
#     macs, params = profile(model, inputs=((image_tensor, tabular_tensor), ), verbose=False)
#     macs, params = clever_format([macs, params], "%.3f")
#     print(f"HyperFusion_AD:{macs}, {params}")
#     print("params", sum(p.numel() for p in model.parameters() if p.requires_grad))
#     # torch.save(model.state_dict(), 'test.pth')
#
#
#     # 测试推理
#     model.eval()
#     with torch.no_grad():
#         output = model((image_tensor, tabular_tensor))
#         print("Output shape:", output.shape)
#     """
#     device cuda
#     input_variance weights initialization
#     HyperFusion_AD:24.553G, 3.854M
#     params 3854090
#     Output shape: torch.Size([2, 2])
#     """

