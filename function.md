---
date: 2024-03-30T19:34:00
tags:
  - Tools
  - python
---

## 1、找出所有全零数据

说明：

写一个方法，使用nibabel包读取一个文件夹下的所有3D数据，打印内容，。其中有些是黑的，全为零，请找出这部分数据

```python

import os
import numpy as np
import nibabel as nib


def find_black_data_files(folder_path):
    """
    找到指定文件夹中所有全零数据（黑数据）的文件。

    :param folder_path: 文件夹路径
    :return: 全零数据文件的完整路径列表
    """
    black_data_files = []

    # 获取文件夹中所有以 .nii.gz 结尾的文件
    nii_files = [f for f in os.listdir(folder_path) if f.endswith('.nii.gz')]

    for file_name in nii_files:
        file_path = os.path.join(folder_path, file_name)

        try:
            # 使用 nibabel 加载 .nii.gz 文件
            img = nib.load(file_path)
            data = img.get_fdata()  # 获取数据数组

            # 检查是否为全零数据
            if np.all(data == 0):
                print(f"发现全零数据文件: {file_path}")
                black_data_files.append(file_name)

        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")

    return black_data_files


def find_union_of_black_data(folder_path1, folder_path2):
    """
    找到两个文件夹中全零数据（黑数据）文件的并集。

    :param folder_path1: 第一个文件夹路径
    :param folder_path2: 第二个文件夹路径
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path1):
        print(f"文件夹路径 {folder_path1} 不存在！")
        return
    if not os.path.exists(folder_path2):
        print(f"文件夹路径 {folder_path2} 不存在！")
        return

    # 分别找到两个文件夹中的黑数据文件
    black_data_files1 = set(find_black_data_files(folder_path1))
    black_data_files2 = set(find_black_data_files(folder_path2))

    # 取并集
    union_black_data_files = black_data_files1.union(black_data_files2)

    # 输出结果
    if union_black_data_files:
        print("\n以下是两个文件夹中全零数据（黑数据）文件的并集：")

        with open ("black_data_union.txt", "w") as f:
            for file_path in sorted(union_black_data_files):
                f.write(file_path + "\n")
                print(file_path)

    else:
        print("未发现全零数据（黑数据）文件。")

if __name__ == "__main__":
    folder_path1 = "/data3/wangchangmiao/shenxy/ADNI/ADNI1/GM"  # 替换为你的第一个文件夹路径
    folder_path2 = "/data3/wangchangmiao/shenxy/ADNI/ADNI1/WM"  # 替换为你的第二个文件夹路径
    find_union_of_black_data(folder_path1, folder_path2)
```

## 2、 删除对应的数据行

说明：已知有一个txt文件和一个csv文件，请实现：
txt文件的每一行和csv文件的第一列的每一行有相同内容
读取txt文件和csv文件，在csv文件中的第一列，如果txt文件里面有，那么删除这一行
最后导出新的csv文件

```python
import pandas as pd

def remove_matching_rows_pandas(txt_file, csv_file, output_file):
    """
    使用 pandas 从 csv 文件中删除第一列与 txt 文件内容匹配的行，并导出新的 csv 文件。

    :param txt_file: txt 文件路径
    :param csv_file: 原始 csv 文件路径
    :param output_file: 新的 csv 文件路径
    """
    # 读取 txt 文件内容
    with open(txt_file, 'r', encoding='utf-8') as f:
        txt_lines = set(line.strip() for line in f if line.strip())

    # 读取 csv 文件
    df = pd.read_csv(csv_file)

    # 筛选第一列不在 txt 文件中的行
    filtered_df = df[~df.iloc[:, 0].isin(txt_lines)]

    # 导出新的 csv 文件
    filtered_df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"处理完成！新的 csv 文件已导出到 {output_file}")


# 示例用法
txt_file = "black_adni2.txt"
csv_file = "./ADNI2_all.csv"
output_file = "./ADNI2_all_noblack.csv"
remove_matching_rows_pandas(txt_file, csv_file, output_file)
```

## 3、将字典转换为csv

```python
from torch import tensor
import torch
import pandas as pd

# 示例字典
Metrics={
    'accuracy': [tensor(0.9479,), tensor(0.9375,), tensor(0.9205,), tensor(0.9545,), tensor(0.9545,)],
'precision': [tensor(0.9706,), tensor(0.9412,), tensor(0.8649,), tensor(1.,), tensor(0.9143,)],
'recall': [tensor(0.8919,), tensor(0.8889,), tensor(0.9412,), tensor(0.8788,), tensor(0.9697,)],
'balanceAccuracy': [tensor(0.9375,), tensor(0.9278,), tensor(0.9243,), tensor(0.9394,), tensor(0.9576,)],
'Specificity': [tensor(0.9831,), tensor(0.9667,), tensor(0.9074,), tensor(1.,), tensor(0.9455,)],
 'auc': [tensor(0.9601,), tensor(0.9769,), tensor(0.9760,), tensor(0.9526,), tensor(0.9868,)],
 'f1': [tensor(0.9296,), tensor(0.9143,), tensor(0.9014,), tensor(0.9355,), tensor(0.9412,)]
         }



# 将字典中的张量转换为标量，并构造一个新的字典
converted_data = {}
for key, tensor_list in Metrics.items():
    # 将每个张量转换为标量（使用 .item() 方法）
    converted_data[key] = [tensor.item() for tensor in tensor_list]

# 转换为 DataFrame
df = pd.DataFrame(converted_data)

# 导出为 Excel 文件
df.to_excel("metrics_output.xlsx", index=False)

print("Excel 文件已成功导出！")

```

