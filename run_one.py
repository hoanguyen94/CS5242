import os
import json
import time
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import collections

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torchvision
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, log_loss

# Utilities
def set_seed(seed: int = 24) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device(use_gpu: bool) -> torch.device:
    if use_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def load_class_mapping(path: Path) -> Dict[str, str]:
    mapping = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                mapping[parts[0]] = parts[1]
    return mapping

def class_names_from_ds(ds_split):
    # Fallback to ids if map_clsloc.txt is not accessible
    return ds_split.features["label"].names

def load_mini_imagenet(subset: Optional[int] = None) -> Dict:
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    ds = load_dataset("timm/mini-imagenet", download_mode="reuse_cache_if_exists")
    if subset is not None:
        ds = {k: v.select(range(min(subset, len(v)))) for k, v in ds.items()}
    ds["class_names"] = class_names_from_ds(ds["train"])
    return ds

def make_transforms(img_size: int = 224, mean=None, std=None, use_aug: bool = False):
    if mean is None or std is None:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    tensor_only = transforms.Compose([transforms.ToTensor()])
    return None, eval_tf, tensor_only

class HFDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, ds_split, transform):
        self.ds = ds_split
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        img: Image.Image = ex["image"].convert("RGB")
        x = self.transform(img)
        y = ex["label"]
        return x, y

def make_loaders(ds, train_tf, eval_tf, batch_size: int = 128, num_workers: int = 2):
    train_ds = HFDatasetWrapper(ds["train"], eval_tf)
    val_ds = HFDatasetWrapper(ds["validation"], eval_tf)
    test_ds = HFDatasetWrapper(ds["test"], eval_tf)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader

# Model definitions
def _replace_classifier_head(model: nn.Module, num_classes: int) -> nn.Module:
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        last = model.classifier[-1]
        if isinstance(last, nn.Linear):
            model.classifier[-1] = nn.Linear(last.in_features, num_classes)
            return model
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError("Unable to replace classifier head")

def build_backbone(backbone: str = "convnext_tiny", num_classes: int = 100, pretrained: bool = True, device: torch.device = torch.device("cpu")) -> nn.Module:
    model_fn = getattr(torchvision.models, backbone)
    
    def _get_weights_enum(name: str):
        overrides = {
            "convnext_tiny": "ConvNeXt_Tiny_Weights",
            "resnet18": "ResNet18_Weights",
            "resnet34": "ResNet34_Weights",
            "resnet50": "ResNet50_Weights",
            "efficientnet_b0": "EfficientNet_B0_Weights",
            "efficientnet_b1": "EfficientNet_B1_Weights",
        }
        return overrides.get(name)

    kwargs = {}
    if pretrained:
        weights_name = _get_weights_enum(backbone)
        if weights_name and hasattr(torchvision.models, weights_name):
            weights_cls = getattr(torchvision.models, weights_name)
            kwargs["weights"] = weights_cls.DEFAULT
        else:
            kwargs["pretrained"] = True
            
    model = model_fn(**kwargs)
    model = _replace_classifier_head(model, num_classes)
    return model.to(device)

def _extract_backbone_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "forward_features"):
        return model.forward_features(x)
    if hasattr(model, "features"):
        return model.features(x)
    if hasattr(model, "fc") and hasattr(model, "avgpool"):
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        if hasattr(model, "maxpool"):
            x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        return x
    raise ValueError("Unsupported model type for feature extraction.")

@torch.no_grad()
def extract_convnext_features(model: nn.Module, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    feats_list, labels_list = [], []
    t0 = time.time()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        feat_map = _extract_backbone_features(model, x)
        feats = feat_map.mean(dim=(2, 3))
        feats_list.append(feats.cpu().numpy())
        labels_list.append(y.numpy())
    ext_time = time.time() - t0
    feats = np.concatenate(feats_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return feats, labels, ext_time

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

# Classical ML Experiment
def classical_ml_experiment(ds, eval_tf, device, backbone="convnext_tiny", clf_type="logreg", batch_size=256, save_dir=Path("./outputs"), img_size=224):
    ensure_dir(save_dir)
    model = build_backbone(backbone=backbone, num_classes=100, pretrained=True, device=device)
    params_mil = count_params(model) / 1e6
    
    train_loader, val_loader, test_loader = make_loaders(ds, eval_tf, eval_tf, batch_size=batch_size)

    print(f"[{backbone} | {clf_type} @ {img_size}] Extracting features ...")
    X_train, y_train, _ = extract_convnext_features(model, train_loader, device)
    X_val, y_val, _ = extract_convnext_features(model, val_loader, device)
    X_test, y_test, test_ext_time = extract_convnext_features(model, test_loader, device)
    
    if clf_type == "logreg":
        clf = LogisticRegression(max_iter=2000, n_jobs=-1, verbose=0)
    elif clf_type == "linear_svm":
        clf = LinearSVC()
        
    t1 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t1
    
    # Eval
    val_acc = accuracy_score(y_val, clf.predict(X_val))
    
    t_inf_start = time.time()
    test_preds = clf.predict(X_test)
    inf_pred_time = time.time() - t_inf_start
    test_acc = accuracy_score(y_test, test_preds)
    
    if hasattr(clf, "predict_proba"):
        test_probs = clf.predict_proba(X_test)
        test_loss = log_loss(y_test, test_probs)
    else:
        test_loss = float('nan') # LinearSVM
        
    inf_time_ms = (test_ext_time + inf_pred_time) / len(y_test) * 1000

    results = {
        "approach": "classical_ml",
        "backbone": backbone,
        "classifier": clf_type,
        "test_acc": test_acc,
        "val_acc": val_acc,
        "Pretrained Models": True,
        "Number of parameters (mil)": params_mil,
        "Test accuracy (%)": test_acc * 100,
        "Test loss (Cross Entropy)": test_loss,
        "Training Time (seconds)": train_time,
        "Inference Time per Image (ms)": inf_time_ms,
        "GPU": device.type != "cpu",
        "Image Size": img_size
    }
    
    out_path = save_dir / f"classical_ml_{clf_type}_{backbone}_{img_size}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {out_path}")
    return results

# Execution
if __name__ == '__main__':
    seed = 24
    use_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
    save_dir = Path("experiments/classical_ml")
    
    set_seed(seed)
    device = get_device(True)  # Force check for GPU/MPS
    
    print(f"Device: {device}")
    ds = load_mini_imagenet(subset=None)
    
    experiments = [("linear_svm", "convnext_tiny", 224)]#
        ("linear_svm", "convnext_tiny", 224),
        ("linear_svm", "resnet18", 224),
        ("linear_svm", "resnet34", 224),
        ("linear_svm", "resnet50", 224),
        ("linear_svm", "efficientnet_b0", 224),
        ("linear_svm", "efficientnet_b1", 224),
        
        ("logreg", "convnext_tiny", 32),
        ("logreg", "resnet18", 32),
        ("logreg", "resnet34", 32),
        ("logreg", "resnet50", 32),
        ("logreg", "efficientnet_b0", 32),
        ("logreg", "efficientnet_b1", 32)
    ]
    
    for clf_type, backbone, img_size in experiments:
        _, eval_tf, _ = make_transforms(img_size=img_size)
        try:
            classical_ml_experiment(ds=ds, eval_tf=eval_tf, device=device, backbone=backbone, clf_type=clf_type, batch_size=256, save_dir=save_dir, img_size=img_size)
        except Exception as e:
            print(f"Failed {backbone} | {clf_type}: {e}")
