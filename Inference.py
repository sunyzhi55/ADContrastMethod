import torch
from torch.utils.data import DataLoader
import time
from Dataset import *
from Net.ComparisonNet import HFBSurv, Interactive_Multimodal_Fusion_Model
import numpy as np
from PIL import Image  # 处理高质量图片
from sklearn.metrics import roc_curve, auc

from torch.nn import CrossEntropyLoss
from torch.optim import *
from Net import *
from MDL_Net.MDL_Net import generate_model
from RLAD_Net.taad import get_model


from HyperFusionNet.HyperFusion_AD_model import HyperFusion_AD
from vapformer.model_components import thenet
from thop import profile, clever_format

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc


def find_best_model_for_resnet(mri_dir, pet_dir, cli_dir, csv_file, batch_size, model_dir, device, save_image_path, experiment_settings, num_classes=2):
    def evaluate_model(model, data_loader, device):
        """评估单个模型并返回AUC值"""
        model.to(device)
        model.eval()
        all_labels = []
        all_probabilities = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                mri_images = batch.get("mri").to(device)
                pet_images = batch.get("pet").to(device)
                label = batch.get("label").to(device)
                mri_pet_images = torch.concat([mri_images, pet_images], dim=1)
                outputs_logit = model(mri_pet_images)
                prob = torch.softmax(outputs_logit, dim=1)
                prob_positive = prob[:, 1]  # 获取正类的概率
                all_probabilities.extend(prob_positive.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
        fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
        roc_auc = auc(fpr, tpr)
        result_dict = {
            'auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            # 'labels': all_labels,
            # 'probs': all_probabilities
        }
        return result_dict

    """查找并评估所有fold模型，返回最佳模型"""

    # 准备数据集
    dataset = MriPetDataset(mri_dir, pet_dir, cli_dir, csv_file,
                            resize_shape=experiment_settings['shape'], valid_group=experiment_settings['task'])
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 使用Path对象
    model_dir = Path(model_dir)
    # 查找所有pth文件
    model_files = list(model_dir.glob('*fold*.pth'))
    if not model_files:
        raise ValueError(f"No model files found in {model_dir} with pattern '*fold*.pth'")

    # 存储所有折的结果
    all_results = []
    best_dict = {
        "name": "resnet",
        "auc": -1,
        "fpr": None,
        "tpr": None,
        "fold": 0
    }

    # 评估每个模型
    for model_path in model_files:
        # 从文件名中提取fold编号
        filename = model_path.name.lower()
        fold_num = None
        if 'fold1' in filename:
            fold_num = 1
        elif 'fold2' in filename:
            fold_num = 2
        elif 'fold3' in filename:
            fold_num = 3
        elif 'fold4' in filename:
            fold_num = 4
        elif 'fold5' in filename:
            fold_num = 5

        if fold_num is None:
            print(f"Warning: Could not determine fold number for {filename}, skipping")
            continue

        print(f"Evaluating fold {fold_num} model: {model_path.name}")
        model = ResnetMriPet(num_classes=num_classes)
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        result_dict = evaluate_model(model, data_loader, device)
        result_dict['fold'] = fold_num
        all_results.append(result_dict)
        print(f"Fold {fold_num} AUC: {result_dict['auc']:.4f}")

        if result_dict["auc"] > best_dict["auc"]:
            best_dict['auc'] = result_dict['auc']
            best_dict['fpr'] = result_dict['fpr']
            best_dict['tpr'] = result_dict['tpr']
            best_dict["fold"] = fold_num

    print("\nBest model results:")
    print(f"Fold: {best_dict['fold']}")
    print(f"AUC: {best_dict['auc']:.4f}")

    # 绘制所有折的ROC曲线
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Random chance')

    colors = ['blue', 'green', 'red', 'cyan', 'magenta']
    for i, result in enumerate(all_results):
        plt.plot(result['fpr'], result['tpr'],
                 color=colors[i],
                 label=f'Fold {result["fold"]} (AUC = {result["auc"]:.3f})')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Model {best_dict["name"]} ROC Curves for All Folds')
    plt.legend(loc='lower right')

    # 保存图片
    output_path = Path(save_image_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"ROC curves saved to {output_path}")

    return best_dict, all_results

def find_best_model_for_efficientNet(mri_dir, pet_dir, cli_dir, csv_file, batch_size, model_dir, device, save_image_path, experiment_settings, num_classes=2):
    def evaluate_model(model, data_loader, device):
        """评估单个模型并返回AUC值"""
        model.to(device)
        model.eval()
        all_labels = []
        all_probabilities = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                mri_images = batch.get("mri").to(device)
                pet_images = batch.get("pet").to(device)
                label = batch.get("label").to(device)
                mri_pet_images = torch.concat([mri_images, pet_images], dim=1)
                outputs_logit = model(mri_pet_images)
                prob = torch.softmax(outputs_logit, dim=1)
                prob_positive = prob[:, 1]  # 获取正类的概率
                all_probabilities.extend(prob_positive.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
        fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
        roc_auc = auc(fpr, tpr)
        result_dict = {
            'auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            # 'labels': all_labels,
            # 'probs': all_probabilities
        }
        return result_dict

    """查找并评估所有fold模型，返回最佳模型"""

    # 准备数据集
    dataset = MriPetDataset(mri_dir, pet_dir, cli_dir, csv_file,
                            resize_shape=experiment_settings['shape'], valid_group=experiment_settings['task'])
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 使用Path对象
    model_dir = Path(model_dir)
    # 查找所有pth文件
    model_files = list(model_dir.glob('*fold*.pth'))
    if not model_files:
        raise ValueError(f"No model files found in {model_dir} with pattern '*fold*.pth'")

    # 存储所有折的结果
    all_results = []
    best_dict = {
        "name": "efficientNet",
        "auc": -1,
        "fpr": None,
        "tpr": None,
        "fold": 0
    }

    # 评估每个模型
    for model_path in model_files:
        # 从文件名中提取fold编号
        filename = model_path.name.lower()
        fold_num = None
        if 'fold1' in filename:
            fold_num = 1
        elif 'fold2' in filename:
            fold_num = 2
        elif 'fold3' in filename:
            fold_num = 3
        elif 'fold4' in filename:
            fold_num = 4
        elif 'fold5' in filename:
            fold_num = 5

        if fold_num is None:
            print(f"Warning: Could not determine fold number for {filename}, skipping")
            continue

        print(f"Evaluating fold {fold_num} model: {model_path.name}")
        model = EfficientNetMriPet(num_classes=num_classes)
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        result_dict = evaluate_model(model, data_loader, device)
        result_dict['fold'] = fold_num
        all_results.append(result_dict)
        print(f"Fold {fold_num} AUC: {result_dict['auc']:.4f}")

        if result_dict["auc"] > best_dict["auc"]:
            best_dict['auc'] = result_dict['auc']
            best_dict['fpr'] = result_dict['fpr']
            best_dict['tpr'] = result_dict['tpr']
            best_dict["fold"] = fold_num

    # 绘制所有折的ROC曲线
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Random chance')

    colors = ['blue', 'green', 'red', 'cyan', 'magenta']
    for i, result in enumerate(all_results):
        plt.plot(result['fpr'], result['tpr'],
                 color=colors[i],
                 label=f'Fold {result["fold"]} (AUC = {result["auc"]:.3f})')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Model {best_dict["name"]} ROC Curves for All Folds')
    plt.legend(loc='lower right')

    # 保存图片
    output_path = Path(save_image_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"ROC curves saved to {output_path}")

    return best_dict, all_results

def find_best_model_for_vit(mri_dir, pet_dir, cli_dir, csv_file, batch_size, model_dir, device, save_image_path, experiment_settings, num_classes=2):
    def evaluate_model(model, data_loader, device):
        """评估单个模型并返回AUC值"""
        model.to(device)
        model.eval()
        all_labels = []
        all_probabilities = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                mri_images = batch.get("mri").to(device)
                pet_images = batch.get("pet").to(device)
                label = batch.get("label").to(device)
                mri_pet_images = torch.concat([mri_images, pet_images], dim=1)
                outputs_logit = model(mri_pet_images)
                prob = torch.softmax(outputs_logit, dim=1)
                prob_positive = prob[:, 1]  # 获取正类的概率
                all_probabilities.extend(prob_positive.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
        fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
        roc_auc = auc(fpr, tpr)
        result_dict = {
            'auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            # 'labels': all_labels,
            # 'probs': all_probabilities
        }
        return result_dict

    """查找并评估所有fold模型，返回最佳模型"""

    # 准备数据集
    dataset = MriPetDataset(mri_dir, pet_dir, cli_dir, csv_file,
                            resize_shape=experiment_settings['shape'], valid_group=experiment_settings['task'])
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 使用Path对象
    model_dir = Path(model_dir)
    # 查找所有pth文件
    model_files = list(model_dir.glob('*fold*.pth'))
    if not model_files:
        raise ValueError(f"No model files found in {model_dir} with pattern '*fold*.pth'")

    # 存储所有折的结果
    all_results = []
    best_dict = {
        "name": "vit",
        "auc": -1,
        "fpr": None,
        "tpr": None,
        "fold": 0
    }

    # 评估每个模型
    for model_path in model_files:
        # 从文件名中提取fold编号
        filename = model_path.name.lower()
        fold_num = None
        if 'fold1' in filename:
            fold_num = 1
        elif 'fold2' in filename:
            fold_num = 2
        elif 'fold3' in filename:
            fold_num = 3
        elif 'fold4' in filename:
            fold_num = 4
        elif 'fold5' in filename:
            fold_num = 5

        if fold_num is None:
            print(f"Warning: Could not determine fold number for {filename}, skipping")
            continue

        print(f"Evaluating fold {fold_num} model: {model_path.name}")
        model = ViTMriPet(num_classes=num_classes)
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        result_dict = evaluate_model(model, data_loader, device)
        result_dict['fold'] = fold_num
        all_results.append(result_dict)
        print(f"Fold {fold_num} AUC: {result_dict['auc']:.4f}")

        if result_dict["auc"] > best_dict["auc"]:
            best_dict['auc'] = result_dict['auc']
            best_dict['fpr'] = result_dict['fpr']
            best_dict['tpr'] = result_dict['tpr']
            best_dict["fold"] = fold_num

    # 绘制所有折的ROC曲线
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Random chance')

    colors = ['blue', 'green', 'red', 'cyan', 'magenta']
    for i, result in enumerate(all_results):
        plt.plot(result['fpr'], result['tpr'],
                 color=colors[i],
                 label=f'Fold {result["fold"]} (AUC = {result["auc"]:.3f})')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Model {best_dict["name"]} ROC Curves for All Folds')
    plt.legend(loc='lower right')

    # 保存图片
    output_path = Path(save_image_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"ROC curves saved to {output_path}")
    return best_dict, all_results

def find_best_model_for_poolformer(mri_dir, pet_dir, cli_dir, csv_file, batch_size, model_dir, device, save_image_path, experiment_settings, num_classes=2):
    def evaluate_model(model, data_loader, device):
        """评估单个模型并返回AUC值"""
        model.to(device)
        model.eval()
        all_labels = []
        all_probabilities = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                mri_images = batch.get("mri").to(device)
                pet_images = batch.get("pet").to(device)
                label = batch.get("label").to(device)
                mri_pet_images = torch.concat([mri_images, pet_images], dim=1)
                outputs_logit = model(mri_pet_images)
                prob = torch.softmax(outputs_logit, dim=1)
                prob_positive = prob[:, 1]  # 获取正类的概率
                all_probabilities.extend(prob_positive.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
        fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
        roc_auc = auc(fpr, tpr)
        result_dict = {
            'auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            # 'labels': all_labels,
            # 'probs': all_probabilities
        }
        return result_dict
    """查找并评估所有fold模型，返回最佳模型"""
    # 准备数据集
    dataset = MriPetDataset(mri_dir, pet_dir, cli_dir, csv_file,
                            resize_shape=experiment_settings['shape'], valid_group=experiment_settings['task'])
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # 使用Path对象
    model_dir = Path(model_dir)
    # 查找所有pth文件
    model_files = list(model_dir.glob('*fold*.pth'))
    if not model_files:
        raise ValueError(f"No model files found in {model_dir} with pattern '*fold*.pth'")
    # 存储所有折的结果
    all_results = []
    best_dict = {
        "name": "poolformer",
        "auc": -1,
        "fpr": None,
        "tpr": None,
        "fold": 0
    }
    # 评估每个模型
    for model_path in model_files:
        # 从文件名中提取fold编号
        filename = model_path.name.lower()
        fold_num = None
        if 'fold1' in filename:
            fold_num = 1
        elif 'fold2' in filename:
            fold_num = 2
        elif 'fold3' in filename:
            fold_num = 3
        elif 'fold4' in filename:
            fold_num = 4
        elif 'fold5' in filename:
            fold_num = 5
        if fold_num is None:
            print(f"Warning: Could not determine fold number for {filename}, skipping")
            continue
        print(f"Evaluating fold {fold_num} model: {model_path.name}")
        model = MetaFormerMriPet(num_classes=num_classes)
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        result_dict = evaluate_model(model, data_loader, device)
        result_dict['fold'] = fold_num
        all_results.append(result_dict)
        print(f"Fold {fold_num} AUC: {result_dict['auc']:.4f}")
        if result_dict["auc"] > best_dict["auc"]:
            best_dict['auc'] = result_dict['auc']
            best_dict['fpr'] = result_dict['fpr']
            best_dict['tpr'] = result_dict['tpr']
            best_dict["fold"] = fold_num
    # 绘制所有折的ROC曲线
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Random chance')
    colors = ['blue', 'green', 'red', 'cyan', 'magenta']
    for i, result in enumerate(all_results):
        plt.plot(result['fpr'], result['tpr'],
                 color=colors[i],
                 label=f'Fold {result["fold"]} (AUC = {result["auc"]:.3f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Model {best_dict["name"]} ROC Curves for All Folds')
    plt.legend(loc='lower right')
    # 保存图片
    output_path = Path(save_image_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"ROC curves saved to {output_path}")
    return best_dict, all_results

def find_best_model_for_nnMamba(mri_dir, pet_dir, cli_dir, csv_file, batch_size, model_dir, device, save_image_path, experiment_settings, num_classes=2):
    def evaluate_model(model, data_loader, device):
        """评估单个模型并返回AUC值"""
        model.to(device)
        model.eval()
        all_labels = []
        all_probabilities = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                mri_images = batch.get("mri").to(device)
                pet_images = batch.get("pet").to(device)
                label = batch.get("label").to(device)
                mri_pet_images = torch.concat([mri_images, pet_images], dim=1)
                outputs_logit = model(mri_pet_images)
                prob = torch.softmax(outputs_logit, dim=1)
                prob_positive = prob[:, 1]  # 获取正类的概率
                all_probabilities.extend(prob_positive.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
        fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
        roc_auc = auc(fpr, tpr)
        result_dict = {
            'auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            # 'labels': all_labels,
            # 'probs': all_probabilities
        }
        return result_dict
    """查找并评估所有fold模型，返回最佳模型"""
    # 准备数据集
    dataset = MriPetDataset(mri_dir, pet_dir, cli_dir, csv_file,
                            resize_shape=experiment_settings['shape'], valid_group=experiment_settings['task'])
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # 使用Path对象
    model_dir = Path(model_dir)
    # 查找所有pth文件
    model_files = list(model_dir.glob('*fold*.pth'))
    if not model_files:
        raise ValueError(f"No model files found in {model_dir} with pattern '*fold*.pth'")
    # 存储所有折的结果
    all_results = []
    best_dict = {
        "name": "nnMamba",
        "auc": -1,
        "fpr": None,
        "tpr": None,
        "fold": 0
    }
    # 评估每个模型
    for model_path in model_files:
        # 从文件名中提取fold编号
        filename = model_path.name.lower()
        fold_num = None
        if 'fold1' in filename:
            fold_num = 1
        elif 'fold2' in filename:
            fold_num = 2
        elif 'fold3' in filename:
            fold_num = 3
        elif 'fold4' in filename:
            fold_num = 4
        elif 'fold5' in filename:
            fold_num = 5
        if fold_num is None:
            print(f"Warning: Could not determine fold number for {filename}, skipping")
            continue
        print(f"Evaluating fold {fold_num} model: {model_path.name}")
        model = nnMambaMriPet(num_classes=num_classes)
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        result_dict = evaluate_model(model, data_loader, device)
        result_dict['fold'] = fold_num
        all_results.append(result_dict)
        print(f"Fold {fold_num} AUC: {result_dict['auc']:.4f}")
        if result_dict["auc"] > best_dict["auc"]:
            best_dict['auc'] = result_dict['auc']
            best_dict['fpr'] = result_dict['fpr']
            best_dict['tpr'] = result_dict['tpr']
            best_dict["fold"] = fold_num
    # 绘制所有折的ROC曲线
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Random chance')
    colors = ['blue', 'green', 'red', 'cyan', 'magenta']
    for i, result in enumerate(all_results):
        plt.plot(result['fpr'], result['tpr'],
                 color=colors[i],
                 label=f'Fold {result["fold"]} (AUC = {result["auc"]:.3f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Model {best_dict["name"]} ROC Curves for All Folds')
    plt.legend(loc='lower right')
    # 保存图片
    output_path = Path(save_image_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"ROC curves saved to {output_path}")
    return best_dict, all_results

def find_best_model_for_VAPL(mri_dir, pet_dir, cli_dir, csv_file, batch_size, model_dir, device, save_image_path, experiment_settings, num_classes=2):
    def evaluate_model(model, data_loader, device):
        """评估单个模型并返回AUC值"""
        model.to(device)
        model.eval()
        all_labels = []
        all_probabilities = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                mri_images = batch.get("mri").to(device)
                cli_tab = batch.get("clinical").to(device)
                label = batch.get("label").to(device)
                input_content = (mri_images, cli_tab)
                outputs_logit = model(input_content)
                prob = torch.softmax(outputs_logit, dim=1)
                prob_positive = prob[:, 1]  # 获取正类的概率
                all_probabilities.extend(prob_positive.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
        fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
        roc_auc = auc(fpr, tpr)
        result_dict = {
            'auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            # 'labels': all_labels,
            # 'probs': all_probabilities
        }
        return result_dict
    """查找并评估所有fold模型，返回最佳模型"""
    # 准备数据集
    dataset = MriCliDataset(mri_dir, pet_dir, cli_dir, csv_file,
                            resize_shape=experiment_settings['shape'], valid_group=experiment_settings['task'])
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # 使用Path对象
    model_dir = Path(model_dir)
    # 查找所有pth文件
    model_files = list(model_dir.glob('*fold*.pth'))
    if not model_files:
        raise ValueError(f"No model files found in {model_dir} with pattern '*fold*.pth'")
    # 存储所有折的结果
    all_results = []
    best_dict = {
        "name": "VAPL",
        "auc": -1,
        "fpr": None,
        "tpr": None,
        "fold": 0
    }
    # 评估每个模型
    for model_path in model_files:
        # 从文件名中提取fold编号
        filename = model_path.name.lower()
        fold_num = None
        if 'fold1' in filename:
            fold_num = 1
        elif 'fold2' in filename:
            fold_num = 2
        elif 'fold3' in filename:
            fold_num = 3
        elif 'fold4' in filename:
            fold_num = 4
        elif 'fold5' in filename:
            fold_num = 5
        if fold_num is None:
            print(f"Warning: Could not determine fold number for {filename}, skipping")
            continue
        print(f"Evaluating fold {fold_num} model: {model_path.name}")
        model = thenet(input_size=[32 * 42 * 32, 16 * 21 * 16, 8 * 10 * 8, 4 * 5 * 4],dims=[32, 64, 128, 256],
                           depths=[3, 3, 3, 3], num_heads=8, in_channels=1,
                           num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        result_dict = evaluate_model(model, data_loader, device)
        result_dict['fold'] = fold_num
        all_results.append(result_dict)
        print(f"Fold {fold_num} AUC: {result_dict['auc']:.4f}")
        if result_dict["auc"] > best_dict["auc"]:
            best_dict['auc'] = result_dict['auc']
            best_dict['fpr'] = result_dict['fpr']
            best_dict['tpr'] = result_dict['tpr']
            best_dict["fold"] = fold_num
    # 绘制所有折的ROC曲线
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Random chance')
    colors = ['blue', 'green', 'red', 'cyan', 'magenta']
    for i, result in enumerate(all_results):
        plt.plot(result['fpr'], result['tpr'],
                 color=colors[i],
                 label=f'Fold {result["fold"]} (AUC = {result["auc"]:.3f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Model {best_dict["name"]} ROC Curves for All Folds')
    plt.legend(loc='lower right')
    # 保存图片
    output_path = Path(save_image_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"ROC curves saved to {output_path}")
    return best_dict, all_results

def find_best_model_for_hyperfusionNet(mri_dir, pet_dir, cli_dir, csv_file, batch_size, model_dir, device, save_image_path, experiment_settings, num_classes=2):
    def evaluate_model(model, data_loader, device):
        """评估单个模型并返回AUC值"""
        model.to(device)
        model.eval()
        all_labels = []
        all_probabilities = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                mri_images = batch.get("mri").to(device)
                cli_tab = batch.get("clinical").to(device)
                label = batch.get("label").to(device)
                input_content = (mri_images, cli_tab)
                outputs_logit = model(input_content)
                prob = torch.softmax(outputs_logit, dim=1)
                prob_positive = prob[:, 1]  # 获取正类的概率
                all_probabilities.extend(prob_positive.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
        fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
        roc_auc = auc(fpr, tpr)
        result_dict = {
            'auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            # 'labels': all_labels,
            # 'probs': all_probabilities
        }
        return result_dict
    """查找并评估所有fold模型，返回最佳模型"""
    # 准备数据集
    dataset = MriCliDataset(mri_dir, pet_dir, cli_dir, csv_file,
                            resize_shape=experiment_settings['shape'], valid_group=experiment_settings['task'])
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # 使用Path对象
    model_dir = Path(model_dir)
    # 查找所有pth文件
    model_files = list(model_dir.glob('*fold*.pth'))
    if not model_files:
        raise ValueError(f"No model files found in {model_dir} with pattern '*fold*.pth'")
    # 存储所有折的结果
    all_results = []
    best_dict = {
        "name": "hyperFusionNet",
        "auc": -1,
        "fpr": None,
        "tpr": None,
        "fold": 0
    }
    # 评估每个模型
    for model_path in model_files:
        # 从文件名中提取fold编号
        filename = model_path.name.lower()
        fold_num = None
        if 'fold1' in filename:
            fold_num = 1
        elif 'fold2' in filename:
            fold_num = 2
        elif 'fold3' in filename:
            fold_num = 3
        elif 'fold4' in filename:
            fold_num = 4
        elif 'fold5' in filename:
            fold_num = 5
        if fold_num is None:
            print(f"Warning: Could not determine fold number for {filename}, skipping")
            continue
        print(f"Evaluating fold {fold_num} model: {model_path.name}")
        model = HyperFusion_AD(train_loader=data_loader, GPU=True, n_outputs=num_classes).to(device)
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        result_dict = evaluate_model(model, data_loader, device)
        result_dict['fold'] = fold_num
        all_results.append(result_dict)
        print(f"Fold {fold_num} AUC: {result_dict['auc']:.4f}")
        if result_dict["auc"] > best_dict["auc"]:
            best_dict['auc'] = result_dict['auc']
            best_dict['fpr'] = result_dict['fpr']
            best_dict['tpr'] = result_dict['tpr']
            best_dict["fold"] = fold_num
    # 绘制所有折的ROC曲线
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Random chance')
    colors = ['blue', 'green', 'red', 'cyan', 'magenta']
    for i, result in enumerate(all_results):
        plt.plot(result['fpr'], result['tpr'],
                 color=colors[i],
                 label=f'Fold {result["fold"]} (AUC = {result["auc"]:.3f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Model {best_dict["name"]} ROC Curves for All Folds')
    plt.legend(loc='lower right')
    # 保存图片
    output_path = Path(save_image_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"ROC curves saved to {output_path}")
    return best_dict, all_results

def find_best_model_for_IMF(mri_dir, pet_dir, cli_dir, csv_file, batch_size, model_dir, device, save_image_path, experiment_settings, num_classes=2):
    def evaluate_model(model, data_loader, device):
        """评估单个模型并返回AUC值"""
        model.to(device)
        model.eval()
        all_labels = []
        all_probabilities = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                mri_images = batch.get("mri").to(device)
                pet_images = batch.get("pet").to(device)
                cli_tab = batch.get("clinical").to(device)
                label = batch.get("label").to(device)
                label_2d = batch.get("label_2d").to(device)
                outputs = model(mri_images, pet_images, cli_tab)
                prob = (outputs[0] + outputs[1] + outputs[2] + outputs[3]) / 4.0
                prob_positive = prob[:, 1]  # 获取正类的概率
                all_probabilities.extend(prob_positive.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
        fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
        roc_auc = auc(fpr, tpr)
        result_dict = {
            'auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            # 'labels': all_labels,
            # 'probs': all_probabilities
        }
        return result_dict
    """查找并评估所有fold模型，返回最佳模型"""
    # 准备数据集
    dataset = MriPetCliDatasetWithTowLabel(mri_dir, pet_dir, cli_dir, csv_file,
                            resize_shape=experiment_settings['shape'], valid_group=experiment_settings['task'])
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # 使用Path对象
    model_dir = Path(model_dir)
    # 查找所有pth文件
    model_files = list(model_dir.glob('*fold*.pth'))
    if not model_files:
        raise ValueError(f"No model files found in {model_dir} with pattern '*fold*.pth'")
    # 存储所有折的结果
    all_results = []
    best_dict = {
        "name": "IMF",
        "auc": -1,
        "fpr": None,
        "tpr": None,
        "fold": 0
    }
    # 评估每个模型
    for model_path in model_files:
        # 从文件名中提取fold编号
        filename = model_path.name.lower()
        fold_num = None
        if 'fold1' in filename:
            fold_num = 1
        elif 'fold2' in filename:
            fold_num = 2
        elif 'fold3' in filename:
            fold_num = 3
        elif 'fold4' in filename:
            fold_num = 4
        elif 'fold5' in filename:
            fold_num = 5
        if fold_num is None:
            print(f"Warning: Could not determine fold number for {filename}, skipping")
            continue
        print(f"Evaluating fold {fold_num} model: {model_path.name}")
        model = Interactive_Multimodal_Fusion_Model(num_classes=num_classes)
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        result_dict = evaluate_model(model, data_loader, device)
        result_dict['fold'] = fold_num
        all_results.append(result_dict)
        print(f"Fold {fold_num} AUC: {result_dict['auc']:.4f}")
        if result_dict["auc"] > best_dict["auc"]:
            best_dict['auc'] = result_dict['auc']
            best_dict['fpr'] = result_dict['fpr']
            best_dict['tpr'] = result_dict['tpr']
            best_dict["fold"] = fold_num
    # 绘制所有折的ROC曲线
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Random chance')
    colors = ['blue', 'green', 'red', 'cyan', 'magenta']
    for i, result in enumerate(all_results):
        plt.plot(result['fpr'], result['tpr'],
                 color=colors[i],
                 label=f'Fold {result["fold"]} (AUC = {result["auc"]:.3f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Model {best_dict["name"]} ROC Curves for All Folds')
    plt.legend(loc='lower right')
    # 保存图片
    output_path = Path(save_image_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"ROC curves saved to {output_path}")
    return best_dict, all_results

def find_best_model_for_HFBSurv(mri_dir, pet_dir, cli_dir, csv_file, batch_size, model_dir, device, save_image_path, experiment_settings, num_classes=2):
    def evaluate_model(model, data_loader, device):
        """评估单个模型并返回AUC值"""
        model.to(device)
        model.eval()
        all_labels = []
        all_probabilities = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                mri_images = batch.get("mri").to(device)
                pet_images = batch.get("pet").to(device)
                cli_tab = batch.get("clinical").to(device)
                label = batch.get("label").to(device)
                outputs_logit = model(mri_images, pet_images, cli_tab)
                prob = torch.softmax(outputs_logit, dim=1)
                prob_positive = prob[:, 1]  # 获取正类的概率
                all_probabilities.extend(prob_positive.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
        fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
        roc_auc = auc(fpr, tpr)
        result_dict = {
            'auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            # 'labels': all_labels,
            # 'probs': all_probabilities
        }
        return result_dict
    """查找并评估所有fold模型，返回最佳模型"""
    # 准备数据集
    dataset = MriPetCliDataset(mri_dir, pet_dir, cli_dir, csv_file,
                            resize_shape=experiment_settings['shape'], valid_group=experiment_settings['task'])
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    # 使用Path对象
    model_dir = Path(model_dir)
    # 查找所有pth文件
    model_files = list(model_dir.glob('*fold*.pth'))
    if not model_files:
        raise ValueError(f"No model files found in {model_dir} with pattern '*fold*.pth'")
    # 存储所有折的结果
    all_results = []
    best_dict = {
        "name": "HFBSuv",
        "auc": -1,
        "fpr": None,
        "tpr": None,
        "fold": 0
    }
    # 评估每个模型
    for model_path in model_files:
        # 从文件名中提取fold编号
        filename = model_path.name.lower()
        fold_num = None
        if 'fold1' in filename:
            fold_num = 1
        elif 'fold2' in filename:
            fold_num = 2
        elif 'fold3' in filename:
            fold_num = 3
        elif 'fold4' in filename:
            fold_num = 4
        elif 'fold5' in filename:
            fold_num = 5
        if fold_num is None:
            print(f"Warning: Could not determine fold number for {filename}, skipping")
            continue
        print(f"Evaluating fold {fold_num} model: {model_path.name}")
        model = HFBSurv(num_classes=num_classes)
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        result_dict = evaluate_model(model, data_loader, device)
        result_dict['fold'] = fold_num
        all_results.append(result_dict)
        print(f"Fold {fold_num} AUC: {result_dict['auc']:.4f}")
        if result_dict["auc"] > best_dict["auc"]:
            best_dict['auc'] = result_dict['auc']
            best_dict['fpr'] = result_dict['fpr']
            best_dict['tpr'] = result_dict['tpr']
            best_dict["fold"] = fold_num
    # 绘制所有折的ROC曲线
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Random chance')
    colors = ['blue', 'green', 'red', 'cyan', 'magenta']
    for i, result in enumerate(all_results):
        plt.plot(result['fpr'], result['tpr'],
                 color=colors[i],
                 label=f'Fold {result["fold"]} (AUC = {result["auc"]:.3f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Model {best_dict["name"]} ROC Curves for All Folds')
    plt.legend(loc='lower right')
    # 保存图片
    output_path = Path(save_image_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"ROC curves saved to {output_path}")
    return best_dict, all_results

def find_best_model_for_ITCFN(mri_dir, pet_dir, cli_dir, csv_file, batch_size, model_dir, device, save_image_path, experiment_settings, num_classes=2):
    def evaluate_model(model, data_loader, device):
        """评估单个模型并返回AUC值"""
        model.to(device)
        model.eval()
        all_labels = []
        all_probabilities = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                mri_images = batch.get("mri").to(device)
                pet_images = batch.get("pet").to(device)
                cli_tab = batch.get("clinical").to(device)
                label = batch.get("label").to(device)
                _, _, _, outputs_logit = model(mri_images, pet_images, cli_tab)
                prob = torch.softmax(outputs_logit, dim=1)
                prob_positive = prob[:, 1]  # 获取正类的概率
                all_probabilities.extend(prob_positive.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
        fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
        roc_auc = auc(fpr, tpr)
        result_dict = {
            'auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            # 'labels': all_labels,
            # 'probs': all_probabilities
        }
        return result_dict
    """查找并评估所有fold模型，返回最佳模型"""
    # 准备数据集
    dataset = MriPetCliDataset(mri_dir, pet_dir, cli_dir, csv_file,
                            resize_shape=experiment_settings['shape'], valid_group=experiment_settings['task'])
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # 使用Path对象
    model_dir = Path(model_dir)
    # 查找所有pth文件
    model_files = list(model_dir.glob('*fold*.pth'))
    if not model_files:
        raise ValueError(f"No model files found in {model_dir} with pattern '*fold*.pth'")
    # 存储所有折的结果
    all_results = []
    best_dict = {
        "name": "ITCFN",
        "auc": -1,
        "fpr": None,
        "tpr": None,
        "fold": 0
    }
    # 评估每个模型
    for model_path in model_files:
        # 从文件名中提取fold编号
        filename = model_path.name.lower()
        fold_num = None
        if 'fold1' in filename:
            fold_num = 1
        elif 'fold2' in filename:
            fold_num = 2
        elif 'fold3' in filename:
            fold_num = 3
        elif 'fold4' in filename:
            fold_num = 4
        elif 'fold5' in filename:
            fold_num = 5
        if fold_num is None:
            print(f"Warning: Could not determine fold number for {filename}, skipping")
            continue
        print(f"Evaluating fold {fold_num} model: {model_path.name}")
        model = Triple_model_CoAttention_Fusion(num_classes=num_classes)
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        result_dict = evaluate_model(model, data_loader, device)
        result_dict['fold'] = fold_num
        all_results.append(result_dict)
        print(f"Fold {fold_num} AUC: {result_dict['auc']:.4f}")
        if result_dict["auc"] > best_dict["auc"]:
            best_dict['auc'] = result_dict['auc']
            best_dict['fpr'] = result_dict['fpr']
            best_dict['tpr'] = result_dict['tpr']
            best_dict["fold"] = fold_num
    # 绘制所有折的ROC曲线
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Random chance')
    colors = ['blue', 'green', 'red', 'cyan', 'magenta']
    for i, result in enumerate(all_results):
        plt.plot(result['fpr'], result['tpr'],
                 color=colors[i],
                 label=f'Fold {result["fold"]} (AUC = {result["auc"]:.3f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Model {best_dict["name"]} ROC Curves for All Folds')
    plt.legend(loc='lower right')
    # 保存图片
    output_path = Path(save_image_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"ROC curves saved to {output_path}")
    return best_dict, all_results


def plot_all_models_roc(all_model_best, save_path='./all_models_roc_curves.png'):
    """
    绘制所有模型的ROC曲线并保存为图片

    参数:
        all_model_best: list of dict, 每个字典包含模型信息和ROC数据
        save_path: str, 图片保存路径
    """
    plt.figure(figsize=(10, 8))

    # 绘制对角线参考线
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random chance')

    # 定义颜色和线型
    colors = plt.cm.get_cmap('tab20', len(all_model_best))
    # line_styles = ['-', '--', ':', '-.']

    for i, model_data in enumerate(all_model_best):
        # 确保fpr和tpr是numpy数组
        fpr = np.array(model_data['fpr'])
        tpr = np.array(model_data['tpr'])

        # 绘制ROC曲线
        plt.plot(fpr, tpr,
                 color=colors(i),
                 # linestyle=line_styles[i % len(line_styles)],
                 linewidth=2,
                 label=f"{model_data['name']} (Fold {model_data['fold']}, AUC={model_data['auc']:.3f})")

    # 美化图形
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison of All Models', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)

    # 保存图片
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curves saved to {save_path}")
    plt.close()

if __name__ == '__main__':
    torch.manual_seed(42)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # 加载数据 - 使用Path对象
    mri_dir = Path('/data3/wangchangmiao/shenxy/ADNI/ADNI1_2/MRI')
    pet_dir = Path('/data3/wangchangmiao/shenxy/ADNI/ADNI1_2/PET')
    cli_dir = Path('./csv/ADNI_Clinical.csv')
    csv_file = Path('./csv/ADNI1_2_ad_mci_cn_validation.csv')
    batch_size = 8
    experiment_settings = {
        'shape': (96, 128, 96),
        'task': ('CN', 'AD')
    }
    all_model_best = []
    # Resnet
    resnet_model_dir = Path(r'/data3/wangchangmiao/shenxy/Code/AD/ad_cn_87/ad_cn/resnet')
    resnet_save_image_path = Path('./roc_curves_resnet.png')
    start_time = time.time()
    # 获取最佳模型和所有结果
    best_dict_resnet, all_results_esnet = find_best_model_for_resnet(
        mri_dir, pet_dir, cli_dir, csv_file, batch_size, resnet_model_dir, device,
        save_image_path = resnet_save_image_path,
        experiment_settings = experiment_settings,
        num_classes=2
    )
    all_model_best.append(best_dict_resnet)
    # efficientNet
    efficientNet_model_dir = Path(r'/data3/wangchangmiao/shenxy/Code/AD/ad_cn_87/ad_cn/efficientNet')
    efficientNet_save_image_path = Path('./roc_curves_efficientNet.png')
    # 获取最佳模型和所有结果
    best_dict_efficientNet, all_results_efficientNet = find_best_model_for_efficientNet(
        mri_dir, pet_dir, cli_dir, csv_file, batch_size, efficientNet_model_dir, device,
        save_image_path = efficientNet_save_image_path,
        experiment_settings=experiment_settings,
        num_classes=2
    )
    all_model_best.append(best_dict_efficientNet)
    # vit
    vit_model_dir = Path(r'/data3/wangchangmiao/shenxy/Code/AD/ad_cn_87/ad_cn/vit')
    vit_save_image_path = Path('./roc_curves_vit.png')
    # 获取最佳模型和所有结果
    best_dict_vit, all_results_vit = find_best_model_for_vit(
        mri_dir, pet_dir, cli_dir, csv_file, batch_size, vit_model_dir, device,
        save_image_path = vit_save_image_path,
        experiment_settings=experiment_settings,
        num_classes=2
    )
    all_model_best.append(best_dict_vit)
    # poolformer
    poolformer_model_dir = Path(r'/data3/wangchangmiao/shenxy/Code/AD/ad_cn_87/ad_cn/poolformer')
    poolformer_save_image_path = Path('./roc_curves_poolformer.png')
    # 获取最佳模型和所有结果
    best_dict_poolformer, all_results_poolformer = find_best_model_for_poolformer(
        mri_dir, pet_dir, cli_dir, csv_file, batch_size, poolformer_model_dir, device,
        save_image_path = poolformer_save_image_path,
        experiment_settings=experiment_settings,
        num_classes=2
    )
    all_model_best.append(best_dict_poolformer)
    # nnMamba
    nnMamba_model_dir = Path(r'/data3/wangchangmiao/shenxy/Code/AD/ad_cn_87/ad_cn/nnMamba')
    nnMamba_save_image_path = Path('./roc_curves_nnMamba.png')
    # 获取最佳模型和所有结果
    best_dict_nnMamba, all_results_nnMamba = find_best_model_for_nnMamba(
        mri_dir, pet_dir, cli_dir, csv_file, batch_size, nnMamba_model_dir, device,
        save_image_path = nnMamba_save_image_path,
        experiment_settings=experiment_settings,
        num_classes=2
    )
    all_model_best.append(best_dict_nnMamba)
    # VAPL
    experiment_settings_for_VAPL = {
        'shape': (96, 128, 96),
        'task': ('CN', 'AD')
    }
    VAPL_model_dir = Path(r'/data3/wangchangmiao/shenxy/Code/AD/ad_cn_87/ad_cn/VAPL')
    VAPL_save_image_path = Path('./roc_curves_VAPL.png')
    # 获取最佳模型和所有结果
    best_dict_VAPL, all_results_VAPL = find_best_model_for_VAPL(
        mri_dir, pet_dir, cli_dir, csv_file, batch_size, VAPL_model_dir, device,
        save_image_path = VAPL_save_image_path,
        experiment_settings=experiment_settings_for_VAPL,
        num_classes=2
    )
    all_model_best.append(best_dict_VAPL)
    # hyperfusionNet
    hyperfusionNet_model_dir = Path(r'/data3/wangchangmiao/shenxy/Code/AD/ad_cn_87/ad_cn/hyperfusionNet')
    hyperfusionNet_save_image_path = Path('./roc_curves_hyperfusionNet.png')
    # 获取最佳模型和所有结果
    best_dict_hyperfusionNet, all_results_hyperfusionNet = find_best_model_for_hyperfusionNet(
        mri_dir, pet_dir, cli_dir, csv_file, batch_size, hyperfusionNet_model_dir, device,
        save_image_path = hyperfusionNet_save_image_path,
        experiment_settings=experiment_settings,
        num_classes=2
    )
    all_model_best.append(best_dict_hyperfusionNet)
    # IMF
    IMF_model_dir = Path(r'/data3/wangchangmiao/shenxy/Code/AD/ad_cn_87/ad_cn/IMF')
    IMF_save_image_path = Path('./roc_curves_IMF.png')
    # 获取最佳模型和所有结果
    best_dict_IMF, all_results_IMF = find_best_model_for_IMF(
        mri_dir, pet_dir, cli_dir, csv_file, batch_size, IMF_model_dir, device,
        save_image_path = IMF_save_image_path,
        experiment_settings=experiment_settings,
        num_classes=2
    )
    all_model_best.append(best_dict_IMF)
    # HFBSurv
    HFBSurv_model_dir = Path(r'/data3/wangchangmiao/shenxy/Code/AD/ad_cn_87/ad_cn/HFBSurv')
    HFBSurv_save_image_path = Path('./roc_curves_HFBSurv.png')
    # 获取最佳模型和所有结果
    best_dict_HFBSurv, all_results_HFBSurv = find_best_model_for_HFBSurv(
        mri_dir, pet_dir, cli_dir, csv_file, batch_size, HFBSurv_model_dir, device,
        save_image_path = HFBSurv_save_image_path,
        experiment_settings=experiment_settings,
        num_classes=2
    )
    all_model_best.append(best_dict_HFBSurv)
    # ITCFN
    ITCFN_model_dir = Path(r'/data3/wangchangmiao/shenxy/Code/AD/ad_cn_87/ad_cn/ITCFN')
    ITCFN_save_image_path = Path('./roc_curves_ITCFN.png')
    # 获取最佳模型和所有结果
    best_dict_ITCFN, all_results_ITCFN = find_best_model_for_ITCFN(
        mri_dir, pet_dir, cli_dir, csv_file, batch_size, ITCFN_model_dir, device,
        save_image_path = ITCFN_save_image_path,
        experiment_settings=experiment_settings,
        num_classes=2
    )
    all_model_best.append(best_dict_ITCFN)
    # 调用函数绘制并保存ROC曲线
    plot_all_models_roc(all_model_best, save_path='./roc_comparison.png')
    end_time = time.time()
    use_time = end_time - start_time
    # 计算小时、分钟和秒
    hours = use_time // 3600
    minutes = (use_time % 3600) // 60
    seconds = use_time % 60
    # 打印总训练时间
    times_result = f'Total time: {hours}h {minutes}m {seconds}s'
    print(times_result)
    """
    已知有一个all_model_best为list,里面存放着各种模型的k折交叉验证中最好的指标，
    每个元素的结构为所示        best_dict = {
        "name": "ITCFN",
        "auc": -1,
        "fpr": 0.35,
        "tpr": 0.5,
        "fold": 0
    }
    请你根据all_model_best里的内容，画出所有模型的ROC曲线，放在一张图中，并且保存为图片
    """

