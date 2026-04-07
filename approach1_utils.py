import time
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, log_loss

from utils import ensure_dir, make_loaders

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

def classical_ml_experiment(ds, eval_tf, device, backbone="convnext_tiny", clf_type="logreg", batch_size=256, save_dir=Path("./outputs"), img_size=224):
    ensure_dir(save_dir)
    model = build_backbone(backbone=backbone, num_classes=100, pretrained=True, device=device)
    params_mil = count_params(model) / 1e6
    
    # Check what make_loaders signature expects - in approach1_v2 it expects ds, eval_tf, eval_tf, batch_size=256, num_workers=2
    # we'll use num_workers=2 since approach1.ipynb had 2
    train_loader, val_loader, test_loader = make_loaders(ds, eval_tf, eval_tf, batch_size=batch_size, num_workers=2)

    print(f"[{backbone} | {clf_type} @ {img_size}] Extracting features ...")
    X_train, y_train, _ = extract_convnext_features(model, train_loader, device)
    X_val, y_val, _ = extract_convnext_features(model, val_loader, device)
    X_test, y_test, test_ext_time = extract_convnext_features(model, test_loader, device)
    
    if clf_type == "logreg":
        # We set verbose=0 to cleanly suppress the internal liblinear spam (like "cg reaches trust region boundary")
        clf = LogisticRegression(max_iter=2000, n_jobs=-1, verbose=0)
    elif clf_type == "linear_svm":
        clf = LinearSVC(verbose=0)
        
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
