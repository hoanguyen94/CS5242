import joblib, torch
from src.model import build_backbone

# Reload
clf = joblib.load("path/to/classical_ml_logreg_resnet18_clf.pkl")
model = build_backbone("resnet18", num_classes=100, pretrained=False, device=device)
model.load_state_dict(torch.load("path/to/classical_ml_logreg_resnet18_backbone.pt"))

# Extract features then predict
from src.model import extract_convnext_features
X, _ = extract_convnext_features(model, loader, device)
preds = clf.predict(X)