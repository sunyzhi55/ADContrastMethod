import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np

# 导入 torchmetrics
from torchmetrics import Accuracy, Recall, Precision, Specificity, F1Score, CohenKappa
from torchmetrics import AUROC, MetricCollection, ConfusionMatrix
from tqdm import tqdm

# 假设这些是你项目中的本地模块
from Dataset import *
from Net import *
from MDL_Net.MDL_Net import generate_model
from AweSomeNet.AweNet import AweSomeNet
from HyperFusionNet.HyperFusion_AD_model import HyperFusion_AD
from vapformer.model_components import thenet
from DiamondNet.DiaMond import DiaMond
from DiamondNet.regbn import RegBN
from MultimodalADNet.multimodalNet import MultimodalADNet


# --- 1. 定义 MetricsCalculator 类 ---
class MetricsCalculator:
    """
    一个使用 torchmetrics 计算和打印分类指标的工具类。
    """

    def __init__(self, num_classes, device):
        self.device = device
        self.num_classes = num_classes

        # 定义一个 MetricCollection，方便一次性更新和计算所有指标
        metrics_dict = {
            'Accuracy': Accuracy(task="binary", num_classes=num_classes),
            'Precision': Precision(task="binary", num_classes=num_classes),
            'Recall': Recall(task="binary", num_classes=num_classes),
            'Specificity': Specificity(task="binary", num_classes=num_classes),
            'F1Score': F1Score(task="binary", num_classes=num_classes),
            'CohenKappa': CohenKappa(task="binary", num_classes=num_classes),
            'AUROC': AUROC(task="binary"),
            'ConfusionMatrix': ConfusionMatrix(task="binary", num_classes=num_classes)
        }
        self.metrics = MetricCollection(metrics_dict).to(device)

    def update(self, preds, target):
        """用一批新的预测和标签更新指标状态"""
        self.metrics.update(preds, target)

    def compute(self):
        """计算所有累积批次的最终指标"""
        return self.metrics.compute()

    def reset(self):
        """重置所有指标的状态"""
        self.metrics.reset()

    def print_results(self, results, model_name, fold_num):
        """格式化并打印计算出的指标"""
        print(f"\n--- {model_name} - Fold {fold_num} Results ---")
        print(f"  Accuracy:    {results['Accuracy'].item():.4f}")
        print(f"  Precision:   {results['Precision'].item():.4f}")
        print(f"  Recall:      {results['Recall'].item():.4f}")
        print(f"  Specificity: {results['Specificity'].item():.4f}")
        print(f"  F1 Score:    {results['F1Score'].item():.4f}")
        print(f"  Cohen Kappa: {results['CohenKappa'].item():.4f}")
        print(f"  AUROC:       {results['AUROC'].item():.4f}")
        print("  Confusion Matrix:")
        print(results['ConfusionMatrix'].cpu().numpy())
        print("--------------------------------------")


# --- DiamondNet 的特殊模型封装 ---
class CompleteModel(nn.Module):
    def __init__(self, diamond, regbn_kwargs, num_classes=2, device='cuda:2', checkpoint_path=None):
        super(CompleteModel, self).__init__()
        checkpoint = torch.load(checkpoint_path, map_location=device) if checkpoint_path else None
        body_model = diamond.body_all(
            modality="multi", block_size=32, image_size=128, patch_size=8,
            num_classes=num_classes, channels=1, dim=512, depth=4, heads=8, dropout=0.0, mlp_dim=309
        )
        [m.load_state_dict(checkpoint['model_state_dict'][i]) for i, m in enumerate(body_model)]
        self.model_pet, self.model_mri, self.model_mp = [m.to(device) for m in body_model]
        self.head = diamond.head(block_size=32, image_size=128, num_classes=num_classes, channels=1)
        self.head.load_state_dict(checkpoint['head_state_dict'])
        self.head.to(device)
        self.regbn = RegBN(**regbn_kwargs).to(device)
        self.kwargs_regbn_train = {"is_training": False}

    def forward(self, pet_input, mri_input):
        output_pet = self.model_pet(pet_input)
        output_mri = self.model_mri(mri_input)
        output_pet, output_mri = self.regbn(output_pet, output_mri, **self.kwargs_regbn_train)
        output_mp = self.model_mp(pet_input, mri_input)
        output = (output_pet + output_mri + output_mp) / 3
        output = self.head(output)
        # DiamondNet直接输出分类得分，而不是logits，需要sigmoid转换为概率
        return torch.sigmoid(output.squeeze(1))


# --- 2. 重构后的统一评估函数 ---
def evaluate_model_folds(
        model_name,
        model_class,
        model_dir,
        dataset_class,
        data_paths,
        experiment_settings,
        device,
        batch_size=8,
        num_classes=2,
        model_init_kwargs=None,
        input_type='mri_pet_concat',
        model_file_pattern='*fold*.pth',
        fold_name_pattern='fold'
):
    """
    对一个模型的所有折进行评估，并计算和打印详细的分类指标。

    参数:
        model_name (str): 模型的名称，用于打印。
        model_class: 模型的类定义。
        model_dir (Path): 存放模型权重文件的目录。
        dataset_class: 数据集的类定义。
        data_paths (dict): 包含 'mri_dir', 'pet_dir', 'cli_dir', 'csv_file' 的字典。
        experiment_settings (dict): 包含 'shape', 'task' 的字典。
        device: torch.device。
        batch_size (int): 批处理大小。
        num_classes (int): 分类类别数。
        model_init_kwargs (dict, optional): 模型初始化时需要的额外参数。
        input_type (str): 定义模型如何接收输入数据，例如 'mri_pet_concat', 'mri_pet_cli' 等。
        model_file_pattern (str): 查找模型文件的glob模式。
        fold_name_pattern (str): 从文件名中提取fold编号的模式 ('fold' or 'split')。
    """
    if model_init_kwargs is None:
        model_init_kwargs = {}

    print(f"\n{'=' * 20} Evaluating Model: {model_name} {'=' * 20}")

    # 1. 准备数据集
    dataset_kwargs = {
        'mri_dir': data_paths['mri_dir'],
        'pet_dir': data_paths['pet_dir'],
        'cli_dir': data_paths['cli_dir'],
        'csv_file': data_paths['csv_file'],
        'valid_group': experiment_settings['task']
    }
    # 根据数据集类添加特定参数
    if 'resize_shape' in dataset_class.__init__.__code__.co_varnames:
        dataset_kwargs['resize_shape'] = experiment_settings['shape']
    if 'is_training' in dataset_class.__init__.__code__.co_varnames:
        dataset_kwargs['is_training'] = False

    dataset = dataset_class(**dataset_kwargs)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             drop_last=True if model_name == "HFBSurv" or model_name == "Diamond" else False)

    # 2. 查找模型文件
    model_dir = Path(model_dir)
    model_files = list(model_dir.glob(model_file_pattern))
    if not model_files:
        print(f"Warning: No model files found in {model_dir} with pattern '{model_file_pattern}'")
        return

    # 3. 初始化指标计算器
    metrics_calculator = MetricsCalculator(num_classes, device)

    all_fold_results = []

    # 4. 评估每个折
    for model_path in sorted(model_files):
        # 从文件名中提取fold编号
        filename = model_path.name.lower()
        fold_num = next((i for i in range(1, 6) if f'{fold_name_pattern}{i}' in filename), None)

        if fold_num is None:
            print(f"Warning: Could not determine fold number for {filename}, skipping.")
            continue

        print(f"\nProcessing {model_name} - Fold {fold_num} (File: {model_path.name})")

        # 实例化和加载模型
        if model_name == "Diamond":  # DiamondNet有特殊的加载方式
            regbn_kwargs = {
                'gpu': 0, 'f_num_channels': 128, 'g_num_channels': 128, 'f_layer_dim': [],
                'g_layer_dim': [], 'normalize_input': True, 'normalize_output': True, 'affine': True,
                'sigma_THR': 0.0, 'sigma_MIN': 0.0,
            }
            model = CompleteModel(model_class(), regbn_kwargs, num_classes, device, model_path)
        elif model_name == "HyperFusionNet":
            model = model_class(train_loader=data_loader, GPU=True,**model_init_kwargs)
        else:
            model = model_class(**model_init_kwargs)
            model.load_state_dict(torch.load(str(model_path), map_location=device), strict=False)

        model.to(device)
        model.eval()
        metrics_calculator.reset()
        train_bar = tqdm(data_loader, desc=f"Inference, model:{model_name}, Fold:{fold_num}", unit="batch")

        with torch.no_grad():
            for batch in train_bar:
                label = batch.get("label").to(device)

                # 根据 input_type 准备模型输入
                if input_type == 'mri_pet_concat':
                    mri = batch.get("mri").to(device)
                    pet = batch.get("pet").to(device)
                    model_input = torch.concat([mri, pet], dim=1)
                    outputs_logit = model(model_input)
                elif input_type == 'mri_cli':
                    mri = batch.get("mri").to(device)
                    cli = batch.get("clinical").to(device)
                    model_input = (mri, cli)
                    outputs_logit = model(model_input)
                elif input_type == 'mri_pet_cli':
                    mri = batch.get("mri").to(device)
                    pet = batch.get("pet").to(device)
                    cli = batch.get("clinical").to(device)
                    model_input = (mri, pet, cli)
                    outputs_logit = model(*model_input)
                elif input_type == 'pet_mri_separate':
                    mri = batch.get("mri").to(device)
                    pet = batch.get("pet").to(device)
                    # DiamondNet 特殊处理，其forward直接输出概率
                    probs_positive = model(pet, mri)
                    metrics_calculator.update(probs_positive, label)
                    continue  # DiamondNet不需要后续的softmax处理

                # 特殊模型的输出处理
                if isinstance(outputs_logit, tuple):
                    if model_name == 'MDL':
                        outputs_logit = outputs_logit[0]
                    # elif model_name == 'IMF':
                    #     # IMF返回4个softmax概率，取平均
                    #     outputs_logit = (outputs_logit[0] + outputs_logit[1] + outputs_logit[2] + outputs_logit[
                    #         3]) / 4.0
                    elif model_name == 'ITCFN':
                        outputs_logit = outputs_logit[3]

                # 计算概率
                if model_name == 'IMF':
                    outputs_logit = (outputs_logit[0] + outputs_logit[1] + outputs_logit[2] + outputs_logit[
                        3]) / 4.0
                    prob = outputs_logit  # 已经做过softmax和平均

                else:
                    prob = torch.softmax(outputs_logit, dim=1)
                # print("prob", type(prob))
                probs_positive = prob[:, 1]  # 获取正类的概率
                metrics_calculator.update(probs_positive, label)

        # 计算并打印当前折的指标
        fold_results = metrics_calculator.compute()
        metrics_calculator.print_results(fold_results, model_name, fold_num)
        all_fold_results.append(fold_results)

    # (可选) 计算并打印所有折的平均指标
    if all_fold_results:
        avg_results = {}
        for key in all_fold_results[0].keys():
            if key != 'ConfusionMatrix':
                avg_results[key] = torch.stack([res[key] for res in all_fold_results]).mean().item()

        print(f"\n--- {model_name} - Average Metrics Across All Folds ---")
        for key, value in avg_results.items():
            print(f"  Avg. {key}: {value:.4f}")
        print("=" * 60 + "\n")


if __name__ == '__main__':
    torch.manual_seed(50)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # --- 统一的数据和实验配置 ---
    data_paths = {
        'mri_dir': Path('/data3/wangchangmiao/shenxy/ADNI/AIBL/MRI'),
        'pet_dir': Path('/data3/wangchangmiao/shenxy/ADNI/AIBL/PET'),
        'cli_dir': Path('./csv/AIBL_Clinical.csv'),
        'csv_file': Path('./csv/AIBL_label.csv'),
    }

    experiment_settings = {
        'shape': (96, 128, 96),
        # 'task': ('sMCI', 'pMCI')
        'task': ('AD', 'CN')
    }

    batch_size = 8
    num_classes = 2

    # --- 模型配置列表 ---
    model_configurations = [
        {
            'model_name': 'ResNet',
            'model_class': ResnetMriPet,
            'model_dir': r'/data3/wangchangmiao/shenxy/Code/AD/ad_cn_87/ad_cn/resnet',
            # 'dataset_class': MriPetDataset,
            'dataset_class': AIBLDataset,
            'input_type': 'mri_pet_concat',
            'model_init_kwargs': {'num_classes': num_classes}
        },
        {
            'model_name': 'ViT',
            'model_class': ViTMriPet,
            'model_dir': r'/data3/wangchangmiao/shenxy/Code/AD/ad_cn_87/ad_cn/vit',
            # 'dataset_class': MriPetDataset,
            'dataset_class': AIBLDataset,
            'input_type': 'mri_pet_concat',
            'model_init_kwargs': {'num_classes': num_classes}
        },
        {
            'model_name': 'nnMamba',
            'model_class': nnMambaMriPet,
            'model_dir': r'/data3/wangchangmiao/shenxy/Code/AD/ad_cn_87/ad_cn/nnMamba',
            # 'dataset_class': MriPetDataset,
            'dataset_class': AIBLDataset,
            'input_type': 'mri_pet_concat',
            'model_init_kwargs': {'num_classes': num_classes}
        },
        {
            'model_name': 'VAPL',
            'model_class': thenet,
            'model_dir': r'/data3/wangchangmiao/shenxy/Code/AD/ad_cn_87/ad_cn/VAPL',
            # 'dataset_class': MriCliDataset,
            'dataset_class': AIBLDataset,
            'input_type': 'mri_cli',
            'model_init_kwargs': {
                'input_size': [32 * 42 * 32, 16 * 21 * 16, 8 * 10 * 8, 4 * 5 * 4],
                'dims': [32, 64, 128, 256], 'depths': [3, 3, 3, 3], 'num_heads': 8,
                'in_channels': 1, 'num_classes': num_classes
            }
        },
        {
            'model_name': 'HyperFusionNet',
            'model_class': HyperFusion_AD,
            'model_dir': r'/data3/wangchangmiao/shenxy/Code/AD/ad_cn_87/ad_cn/hyperfusionNet',
            # 'dataset_class': MriCliDataset,
            'dataset_class': AIBLDataset,
            'input_type': 'mri_cli',
            'model_init_kwargs': {'n_outputs': num_classes, 'n_tabular_features':9},

        },
        {
            'model_name': 'MDL',
            'model_class': generate_model,
            'model_dir': r'/data3/wangchangmiao/shenxy/Code/AD/ad_cn_87/ad_cn/MDL',
            # 'dataset_class': MriPetDataset,
            'dataset_class': AIBLDataset,
            'input_type': 'mri_pet_concat',
            'model_init_kwargs': {'model_depth': 18, 'in_planes': 1, 'num_classes': num_classes}
        },
        {
            'model_name': 'HFBSurv',
            'model_class': HFBSurv,
            'model_dir': r'/data3/wangchangmiao/shenxy/Code/AD/ad_cn_87/ad_cn/HFBSurv',
            # 'dataset_class': MriPetCliDataset,
            'dataset_class': AIBLDataset,
            'input_type': 'mri_pet_cli',
            'model_init_kwargs': {'num_classes': num_classes}
        },
        {
            'model_name': 'ITCFN',
            'model_class': Triple_model_CoAttention_Fusion,
            'model_dir': r'/data3/wangchangmiao/shenxy/Code/AD/ad_cn_87/ad_cn/ITCFN',
            # 'dataset_class': MriPetCliDataset,
            'dataset_class': AIBLDataset,
            'input_type': 'mri_pet_cli',
            'model_init_kwargs': {'num_classes': num_classes}
        },
        {
            'model_name': 'MultimodalADNet',
            'model_class': MultimodalADNet,
            'model_dir': r'/data3/wangchangmiao/shenxy/Code/AD/ad_cn_87/ad_cn/MultimodalADNet',
            # 'dataset_class': MriPetCliDataset,
            'dataset_class': AIBLDataset,
            'input_type': 'mri_pet_cli',
            'model_init_kwargs': {'num_classes': num_classes, 'dim': 64, 'tabular_dim':9}
        },
        {
            'model_name': 'TriLightNet',  # AweSomeNet
            'model_class': AweSomeNet,
            'model_dir': r'/data3/wangchangmiao/shenxy/Code/AD/AweSome/AwesomeNetFinal_lr00001/pmci_smci',
            # 'dataset_class': MriPetCliDataset,
            'dataset_class': AIBLDataset,
            'input_type': 'mri_pet_cli',
            'model_init_kwargs': {'num_classes': num_classes}
        },
        {
            'model_name': 'Diamond',
            'model_class': DiaMond,  # Pass the base class, CompleteModel will be used internally
            'model_dir': r'/data3/wangchangmiao/shenxy/Code/AD/DiaMond/ad_cn/',
            # 'dataset_class': MriPetDatasetForDiamond,
            'dataset_class': AIBLDatasetForDiamond,
            'input_type': 'pet_mri_separate',
            'model_init_kwargs': {},
            'model_file_pattern': '*bestval.pt',
            'fold_name_pattern': 'split'
        },
        # 注释掉的模型可以轻松地取消注释以进行评估
        {
            'model_name': 'IMF',
            'model_class': Interactive_Multimodal_Fusion_Model,
            'model_dir': r'/data3/wangchangmiao/shenxy/Code/AD/ad_cn_87/ad_cn/IMF',
            # 'dataset_class': MriPetCliDatasetWithTowLabel,
            'dataset_class': AIBLDataset,
            'input_type': 'mri_pet_cli',
            'model_init_kwargs': {'num_classes': num_classes}
        },
    ]

    start_time = time.time()

    # --- 循环执行评估 ---
    for config in model_configurations:
        evaluate_model_folds(
            model_name=config['model_name'],
            model_class=config['model_class'],
            model_dir=config['model_dir'],
            dataset_class=config['dataset_class'],
            data_paths=data_paths,
            experiment_settings=experiment_settings,
            device=device,
            batch_size=batch_size,
            num_classes=num_classes,
            model_init_kwargs=config.get('model_init_kwargs'),
            input_type=config.get('input_type'),
            model_file_pattern=config.get('model_file_pattern', '*fold*.pth'),
            fold_name_pattern=config.get('fold_name_pattern', 'fold')
        )

    end_time = time.time()
    use_time = end_time - start_time
    hours = use_time // 3600
    minutes = (use_time % 3600) // 60
    seconds = use_time % 60

    print(f'Total evaluation time: {int(hours)}h {int(minutes)}m {seconds:.2f}s')