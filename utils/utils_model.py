import os, json
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def load_checkpoint(ckpt_path):
    """Load checkpoint; support versioned dict or raw state_dict."""
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
        classes = obj.get("classes", None)
        img_size = obj.get("img_size", 256)
        norm = obj.get("normalize", {"mean": IMAGENET_MEAN, "std": IMAGENET_STD})
        mean, std = norm.get("mean", IMAGENET_MEAN), norm.get("std", IMAGENET_STD)
        return sd, classes, img_size, mean, std
    # fallback: raw state_dict
    return obj, None, 256, IMAGENET_MEAN, IMAGENET_STD

def build_model(state_dict, num_classes=None):
    """Rebuild ResNet18 head (with/without Dropout) based on keys in state_dict."""
    head_key = "fc.1.weight" if "fc.1.weight" in state_dict else "fc.weight"
    out_dim = state_dict[head_key].shape[0]
    if num_classes is None:
        num_classes = out_dim

    m = models.resnet18(weights=None)
    if "fc.1.weight" in state_dict:
        m.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(m.fc.in_features, num_classes))
    else:
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    m.load_state_dict(state_dict, strict=True)
    m.eval()
    return m

def build_loader(train_dir, val_dir, img_size, mean, std, batch_size, device):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = datasets.ImageFolder(train_dir, transform=tf)
    val_ds   = datasets.ImageFolder(val_dir,   transform=tf)

    filenames = [p for p,_ in train_ds.samples] + [p for p,_ in val_ds.samples]
    classes = train_ds.classes  # ImageFolder ensures same order with val
    full = torch.utils.data.ConcatDataset([train_ds, val_ds])

    loader = DataLoader(
        full, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(device == "cuda")
    )
    return loader, classes, filenames