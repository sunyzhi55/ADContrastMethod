import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import torch
from torch.utils.data import DataLoader
from torchmetrics import AUROC, Accuracy, CohenKappa, ConfusionMatrix, F1Score, MetricCollection, Precision, Recall, Specificity

from model_object import models


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    mri_dir: str
    pet_dir: str
    cli_dir: str
    csv_file: str
    weights_dir: str


def _now_tag() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _as_float(x: Any) -> float:
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[len("module."):]: v for k, v in state_dict.items()}

class MetricsComputer:
    def __init__(
        self,
        device: torch.device,
        num_classes: int,
        task: str,
        average: str,
    ) -> None:
        self.device = device
        self.num_classes = num_classes
        self.task = task
        self.average = average

        self.metric_collection = MetricCollection(
            {
                "confusionMatrix": ConfusionMatrix(num_classes=num_classes, task=task).to(device),
                "Accuracy": Accuracy(num_classes=num_classes, task=task).to(device),
                "Precision": Precision(num_classes=num_classes, task=task, average=average).to(device),
                "Recall": Recall(num_classes=num_classes, task=task, average=average).to(device),
                "Specificity": Specificity(num_classes=num_classes, task=task, average=average).to(device),
                "F1": F1Score(num_classes=num_classes, task=task, average=average).to(device),
                "CohenKappa": CohenKappa(num_classes=num_classes, task=task).to(device),
            }
        ).to(device)
        self.auc_metric = AUROC(num_classes=num_classes, task=task).to(device)

    def reset(self) -> None:
        self.metric_collection.reset()
        self.auc_metric.reset()

    def update(self, prob: torch.Tensor, pred: torch.Tensor, label: torch.Tensor) -> None:
        self.metric_collection.forward(pred, label)
        if self.task == "binary":
            prob_positive = prob[:, 1]
        else:
            prob_positive = prob
        self.auc_metric.update(prob_positive, label)

    def compute(self) -> Dict[str, Any]:
        out = self.metric_collection.compute()
        auc = self.auc_metric.compute()
        out_dict = {k: v for k, v in out.items()}
        out_dict["AuRoc"] = auc
        out_dict["BalanceAccuracy"] = (out_dict["Specificity"] + out_dict["Recall"]) / 2.0
        return out_dict


def _infer_prob_and_pred(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    label = batch.get("label")
    if label is None:
        raise ValueError("Batch missing key 'label'")
    label = label.to(device)

    mri = batch.get("mri") if "mri" in batch else None
    pet = batch.get("pet") if "pet" in batch else None
    cli = batch.get("clinical") if "clinical" in batch else None

    if mri is not None:
        mri = mri.to(device)
    if pet is not None:
        pet = pet.to(device)
    if cli is not None:
        cli = cli.to(device)


    # Most models in this repo take (mri, pet, cli)
    if mri is not None and pet is not None and cli is not None:
        outputs = model(mri, pet, cli)
    elif mri is not None and cli is not None and pet is None:
        outputs = model(mri, cli)
    elif mri is not None and pet is not None and cli is None:
        # print(f"model class name: {model.__class__.__name__}")
        if model.__class__.__name__ in ("nnMambaMriPet", "MDL_Net"):
            mri_pet_images = torch.concat([mri, pet], dim=1)
            outputs = model(mri_pet_images)
        else:
            outputs = model(mri, pet)
    elif mri is not None and pet is None and cli is None:
        outputs = model(mri)
    else:
        raise ValueError(f"Unsupported batch keys: {list(batch.keys())}")

    # IHFNet/IMF style: list of 4 class-prob tensors (already sigmoid)
    if isinstance(outputs, (list, tuple)) and len(outputs) == 4 and all(torch.is_tensor(o) and o.dim() == 2 for o in outputs):
        prob = (outputs[0] + outputs[1] + outputs[2] + outputs[3]) / 4.0
    # Some models return (features..., logits)
    elif isinstance(outputs, (list, tuple)):
        logits = None
        for item in reversed(outputs):
            if torch.is_tensor(item) and item.dim() == 2:
                logits = item
                break
        if logits is None:
            raise ValueError(f"Cannot find logits tensor in model outputs: {type(outputs)}")
        prob = torch.softmax(logits, dim=1)
    # Plain logits
    elif torch.is_tensor(outputs) and outputs.dim() == 2:
        prob = torch.softmax(outputs, dim=1)
    # MoE dict style
    elif isinstance(outputs, dict) and "output" in outputs and torch.is_tensor(outputs["output"]):
        prob = torch.softmax(outputs["output"], dim=1)
    else:
        raise ValueError(f"Unsupported model outputs type: {type(outputs)}")

    _, pred = torch.max(prob, dim=1)
    return prob, pred, label


def _evaluate_one_fold(
    model_index: str,
    fold: int,
    weights_dir: Path,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    exp = models[model_index]
    num_classes = len(exp["task"])

    model_cls = exp["Model"]
    # model = model_cls(n_outputs=num_classes)
    if model_index == 'MDL':
        model = model_cls(model_depth=18, in_planes=1, num_classes=num_classes)
    else:
        model = model_cls(num_classes=num_classes)
    
    model = model.to(device)

    weights_path = weights_dir / f"{exp['Name']}_best_model_fold{fold}.pth"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing fold-{fold} weights: {weights_path}")

    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and all(torch.is_tensor(v) for v in state.values()):
        state_dict = _strip_module_prefix(state)
        model.load_state_dict(state_dict, strict=True)
    else:
        raise ValueError(f"Unexpected checkpoint content type for {weights_path}: {type(state)}")

    model.eval()

    task = "multiclass" if num_classes > 2 else "binary"
    average = "macro" if num_classes > 2 else "micro"
    metrics = MetricsComputer(device=device, num_classes=num_classes, task=task, average=average)
    metrics.reset()

    with torch.no_grad():
        for batch in dataloader:
            if not isinstance(batch, dict):
                raise ValueError(
                    f"This evaluator expects dict batches, got {type(batch)}. "
                    "If you need tuple-batch datasets, we can extend this safely."
                )
            prob, pred, label = _infer_prob_and_pred(model, batch, device)
            metrics.update(prob=prob, pred=pred, label=label)

    computed = metrics.compute()
    # Convert tensors to python types for JSON/printing
    result = {
        "Accuracy": _as_float(computed["Accuracy"]),
        "Precision": _as_float(computed["Precision"]),
        "Recall": _as_float(computed["Recall"]),
        "Specificity": _as_float(computed["Specificity"]),
        "BalanceAccuracy": _as_float(computed["BalanceAccuracy"]),
        "F1": _as_float(computed["F1"]),
        "CohenKappa": _as_float(computed["CohenKappa"]),
        "AuRoc": _as_float(computed["AuRoc"]),
        "confusionMatrix": computed["confusionMatrix"].detach().cpu().tolist(),
    }
    return result


def _mean_metrics(per_fold: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "Accuracy",
        "Precision",
        "Recall",
        "Specificity",
        "BalanceAccuracy",
        "F1",
        "CohenKappa",
        "AuRoc",
    ]
    out: Dict[str, Any] = {}
    for k in keys:
        out[k] = float(sum(m[k] for m in per_fold) / len(per_fold))
    return out


def _print_metrics(title: str, metrics: Dict[str, Any]) -> None:
    print(title)
    for k, v in metrics.items():
        if k == "confusionMatrix":
            continue
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")


def _run_direction(
    train_cfg: DatasetConfig,
    test_cfg: DatasetConfig,
    model_indices: List[str],
    n_splits: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    output_dir: Path,
) -> None:
    direction_tag = f"{train_cfg.name}_to_{test_cfg.name}"
    print(f"\n========== {direction_tag} ==========")

    all_results: Dict[str, Any] = {
        "direction": direction_tag,
        "train_weights_dir": train_cfg.weights_dir,
        "test_dataset": {
            "mri_dir": test_cfg.mri_dir,
            "pet_dir": test_cfg.pet_dir,
            "cli_dir": test_cfg.cli_dir,
            "csv_file": test_cfg.csv_file,
        },
        "models": {},
    }

    for model_index in model_indices:
        exp = models[model_index]
        num_classes = len(exp["task"])
        print(f"\n--- Evaluating model: {model_index} ---")
        print(f"exp[task]: {exp['task']}")
        

        dataset_cls = exp["dataset"]
        test_dataset = dataset_cls(
            test_cfg.mri_dir,
            test_cfg.pet_dir,
            test_cfg.cli_dir,
            test_cfg.csv_file,
            resize_shape=exp["shape"],
            valid_group=exp["task"],
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )

        weights_dir = Path(train_cfg.weights_dir)
        per_fold: List[Dict[str, Any]] = []
        print(f"\n--- Model {model_index} (exp name: {exp['Name']}, classes: {num_classes}) ---")

        for fold in range(1, n_splits + 1):
            fold_metrics = _evaluate_one_fold(
                model_index=model_index,
                fold=fold,
                weights_dir=weights_dir,
                dataloader=test_loader,
                device=device,
            )
            per_fold.append(fold_metrics)
            _print_metrics(f"Fold {fold} metrics:", fold_metrics)

        mean_metrics = _mean_metrics(per_fold)
        _print_metrics("Mean over folds:", mean_metrics)

        all_results["models"][model_index] = {
            "exp_name": exp["Name"],
            "per_fold": per_fold,
            "mean": mean_metrics,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"cross_dataset_test_{direction_tag}_{_now_tag()}.json"
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved results to: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-dataset test: load 5-fold weights trained on one dataset, test on the other.")

    parser.add_argument(
        "--direction",
        choices=["adni1_to_adni2", "adni2_to_adni1", "both"],
        default="both",
        help="Test direction.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=["all"],
        help="Model indices from model_object.models, or 'all'. Example: --models IHFNet_With_MLP TriLightNet",
    )
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="./cross_dataset_test_outputs")

    # ADNI1
    parser.add_argument("--adni1_mri_dir", type=str, default='/data3/wangchangmiao/shenxy/ADNI/ADNI1/MRI')
    parser.add_argument("--adni1_pet_dir", type=str, default='/data3/wangchangmiao/shenxy/ADNI/ADNI1/PET')
    parser.add_argument("--adni1_cli_dir", type=str, default='./csv/ADNI_Clinical.csv')
    parser.add_argument("--adni1_csv_file", type=str, default='./csv/ADNI1_match.csv')
    parser.add_argument("--adni1_weights_dir", type=str, default='./checkpoints_2025-12-22_11-05')

    # ADNI2
    parser.add_argument("--adni2_mri_dir", type=str, default='/data3/wangchangmiao/shenxy/ADNI/ADNI2/MRI')
    parser.add_argument("--adni2_pet_dir", type=str, default='/data3/wangchangmiao/shenxy/ADNI/ADNI2/PET')
    parser.add_argument("--adni2_cli_dir", type=str, default='./csv/ADNI_Clinical.csv')
    parser.add_argument("--adni2_csv_file", type=str, default='./csv/ADNI2_match.csv')
    parser.add_argument("--adni2_weights_dir", type=str, default='./checkpoints_2025-12-22_11-10')

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.models == ["all"] or (len(args.models) == 1 and args.models[0].lower() == "all"):
        model_indices = sorted(models.keys())
    else:
        unknown = [m for m in args.models if m not in models]
        if unknown:
            raise ValueError(f"Unknown models: {unknown}. Available: {sorted(models.keys())}")
        model_indices = args.models

    device = torch.device(args.device)

    adni1 = DatasetConfig(
        name="ADNI1",
        mri_dir=args.adni1_mri_dir,
        pet_dir=args.adni1_pet_dir,
        cli_dir=args.adni1_cli_dir,
        csv_file=args.adni1_csv_file,
        weights_dir=args.adni1_weights_dir,
    )
    adni2 = DatasetConfig(
        name="ADNI2",
        mri_dir=args.adni2_mri_dir,
        pet_dir=args.adni2_pet_dir,
        cli_dir=args.adni2_cli_dir,
        csv_file=args.adni2_csv_file,
        weights_dir=args.adni2_weights_dir,
    )

    output_dir = Path(args.output_dir)

    if args.direction in ("adni1_to_adni2", "both"):
        _run_direction(
            train_cfg=adni1,
            test_cfg=adni2,
            model_indices=model_indices,
            n_splits=args.n_splits,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            output_dir=output_dir,
        )

    if args.direction in ("adni2_to_adni1", "both"):
        _run_direction(
            train_cfg=adni2,
            test_cfg=adni1,
            model_indices=model_indices,
            n_splits=args.n_splits,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()

"""
python cross_dataset_test.py --direction both --models IHFNet_With_MLP  --n_splits 5 --batch_size 8 --device cuda:0 
nohup python cross_dataset_test.py --direction both --models MDL  --n_splits 5 --batch_size 8 --device cuda:1  > result_cross_test_MDL.out &

"""
