import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import AdaptiveAvgPool1d, AdaptiveMaxPool1d
from MultimodalADNet.networks import *
from MultimodalADNet.backbones import *


class MultimodalADNet(nn.Module):
    def __init__(
            self,
            dim=64,
            num_classes=2,
            heads_num=4,
            dropout=0.1,
            trans_enc_depth=3,
            tabular_dim=9,
            mri_cnn_channels=None,
            pet_cnn_channels=None,
    ):
        super().__init__()

        # CNN Backbones
        if pet_cnn_channels is None:
            pet_cnn_channels = [16, 32, 64, 128, 128, dim]
        if mri_cnn_channels is None:
            mri_cnn_channels = [32, 64, 128, 256, 256, dim]
        self.mri_cnn = SFCN(channel_number=mri_cnn_channels)
        self.pet_cnn = SFCN(channel_number=pet_cnn_channels)

        # Transformers
        # self.mri_trans = Transformer_IT(
        #     dim=dim,
        #     tabular_dim=tabular_dim,
        #     depth=trans_enc_depth,
        #     heads=heads_num,
        #     dim_head=dim // heads_num,
        #     mlp_dim=dim * 4,
        #     dropout=dropout
        # )

        self.pet_trans = CrossTransformer_IT(
            dim=dim,
            tabular_dim=tabular_dim,
            depth=trans_enc_depth,
            heads=heads_num,
            dim_head=dim // heads_num,
            mlp_dim=dim * 4,
            dropout=dropout
        )

        # Pooling & Classification Heads
        self.gap = nn.Sequential(
            Rearrange('b n d -> b d n'),
            AdaptiveAvgPool1d(1),
            Rearrange('b d n -> b (d n)')
        )
        self.gmp = nn.Sequential(
            Rearrange('b n d -> b d n'),
            AdaptiveMaxPool1d(1),
            Rearrange('b d n -> b (d n)')
        )

        self.mri_cls_head = nn.Linear(dim * 2, num_classes)
        self.pet_cls_head = nn.Linear(dim * 2, num_classes)

    def forward(self, mri, pet, tabular_data):
        """
        Args:
            mri: MRI input shape [batch, 1, H, W, D]
            pet: PET input shape [batch, 1, H, W, D]
            tabular_data: Tabular features of shape [batch, tabular_dim]
        Returns:
            mri_logits: MRI predictions [batch, class_num]
            pet_logits: PET predictions [batch, class_num]
            pet_feats: Processed PET features [batch, 252, 64]
        """
        # Feature Extraction
        mri_feats = self.mri_cnn(mri)  # [2, 64, 6, 7, 6]
        pet_feats = self.pet_cnn(pet)  # [2, 64, 6, 7, 6]

        # Rearrange to token sequences
        mri_feats = rearrange(mri_feats, 'b c h w d -> b (h w d) c')  # [2, 252, 64]
        pet_feats = rearrange(pet_feats, 'b c h w d -> b (h w d) c')  # [2, 252, 64]

        # Transformer Processing
        # mri_feats = self.mri_trans(mri_feats, tabular_data)  # Optional
        pet_feats = self.pet_trans(mri_feats, pet_feats, tabular_data)

        # Classification
        cls_avg = self.gap(pet_feats)  # [2, 64]
        cls_max = self.gmp(pet_feats)  # [2, 64]
        cls_token = torch.cat([cls_avg, cls_max], dim=1)  # [2, 128]

        logits = self.pet_cls_head(cls_token)  # [2, class_num]

        return logits


# 使用示例
if __name__ == "__main__":
    net = MultimodalADNet(dim=64)
    mri = torch.randn(2, 1, 193, 229, 193)
    pet = torch.randn(2, 1, 193, 229, 193)
    tabular = torch.randn(2, 9)

    logits = net(mri, pet, tabular)
    from thop import profile, clever_format
    flops, params = profile(net, inputs=(mri, pet, tabular))
    flops, params = clever_format([flops, params], "%.3f")
    print("FLOPs:", flops, "Params:", params) # FLOPs: 288.943G Params: 4.642M
    print("Logits shape:", logits.shape)  # [2, 2]
