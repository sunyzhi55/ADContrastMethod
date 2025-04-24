# from vapformer.model_components import thenet
# import torch
# model = thenet(
#     input_size=[32 * 42 * 32, 16 * 21 * 16, 8 * 10 * 8, 4 * 5 * 4],
#     dims=[32, 64, 128, 256],
#     depths=[3, 3, 3, 3],
#     num_heads=8,
#     in_channels=1,
#     num_classes=2
# )
# mri_image = torch.randn(1, 1, 96, 128, 96)
# cli_tab = torch.randn(1, 9)
# out = model((mri_image, cli_tab))
# print(out.shape)
from torch import tensor
import torch
import pandas as pd

# # 示例字典
# Metrics={
#     'accuracy': [tensor(0.9479,), tensor(0.9375,), tensor(0.9205,), tensor(0.9545,), tensor(0.9545,)],
# 'precision': [tensor(0.9706,), tensor(0.9412,), tensor(0.8649,), tensor(1.,), tensor(0.9143,)],
# 'recall': [tensor(0.8919,), tensor(0.8889,), tensor(0.9412,), tensor(0.8788,), tensor(0.9697,)],
# 'balanceAccuracy': [tensor(0.9375,), tensor(0.9278,), tensor(0.9243,), tensor(0.9394,), tensor(0.9576,)],
# 'Specificity': [tensor(0.9831,), tensor(0.9667,), tensor(0.9074,), tensor(1.,), tensor(0.9455,)],
#  'auc': [tensor(0.9601,), tensor(0.9769,), tensor(0.9760,), tensor(0.9526,), tensor(0.9868,)],
#  'f1': [tensor(0.9296,), tensor(0.9143,), tensor(0.9014,), tensor(0.9355,), tensor(0.9412,)]
#          }
#
#
#
# # 将字典中的张量转换为标量，并构造一个新的字典
# converted_data = {}
# for key, tensor_list in Metrics.items():
#     # 将每个张量转换为标量（使用 .item() 方法）
#     converted_data[key] = [tensor.item() for tensor in tensor_list]
#
# # 转换为 DataFrame
# df = pd.DataFrame(converted_data)
#
# # 导出为 Excel 文件
# df.to_excel("metrics_output.xlsx", index=False)
#
# print("Excel 文件已成功导出！")
from vapformer.model_components import thenet
model = thenet(input_size=[32 * 42 * 32, 16 * 21 * 16, 8 * 10 * 8, 4 * 5 * 4],
                           dims=[32, 64, 128, 256],
                           depths=[3, 3, 3, 3], num_heads=8, in_channels=1,
                           num_classes=2)
mri_image = torch.randn(1, 1, 96, 128, 96)
cli_tab = torch.randn(1, 9)
out = model((mri_image, cli_tab))
print(out.shape)