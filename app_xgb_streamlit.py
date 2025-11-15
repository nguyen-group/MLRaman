# app_xgb_streamlit.py
import os
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps
import cv2
import xgboost as xgb

# ===================== PATH =====================
CKPT_PTH = r"./model/resnet18_10cls_20251115.pth"    # checkpoint CNN
XGB_PKL  = r"./output/xgb_on_cnn.pkl"                # option pkl backup

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.set_page_config(page_title="MLRaman with CNN+XGBoost", layout="centered")

# ===================== IMAGE UTIL =====================
def preprocess_image_for_cnn(pil_img, img_size, mean, std):
    """Resize → normalize → tensor (ONLY for CNN prediction)."""

    pil_resized = ImageOps.contain(pil_img, (img_size, img_size))

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    tensor = tfm(pil_resized).unsqueeze(0)
    return tensor


# ===================== FEATURE EXTRACTOR =====================
class ResNet18Features(nn.Module):
    def __init__(self, resnet18):
        super().__init__()
        self.features = nn.Sequential(
            resnet18.conv1, resnet18.bn1, resnet18.relu, resnet18.maxpool,
            resnet18.layer1, resnet18.layer2, resnet18.layer3, resnet18.layer4,
            resnet18.avgpool
        )

    def forward(self, x):
        x = self.features(x)
        return torch.flatten(x, 1)

@st.cache_resource
def load_feature_extractor_and_meta(pth_path):
    """
    Load checkpoint .pth: model CNN, classes, img_size, mean/std.
    Use ResNet18 to extract 512-D features (we do not use fully-connected).
    """
    ckpt = torch.load(pth_path, map_location="cpu")

    class_names = ckpt["classes"]
    img_size = ckpt.get("img_size", 224)

    if isinstance(img_size, (tuple, list)):
        img_size = int(img_size[0])

    mean = ckpt.get("normalize", {}).get("mean", [0.485, 0.456, 0.406])
    std  = ckpt.get("normalize", {}).get("std",  [0.229, 0.224, 0.225])

    # build resnet18 model and load weights
    backbone = models.resnet18(weights=None)
    in_feats = backbone.fc.in_features
    # block fc in checkpoint might be nn.Sequential -> map
    backbone.fc = nn.Linear(in_feats, len(class_names))

    state_dict = ckpt["state_dict"]
    # Normalize keys (if previously was fc.1 -> now only fc)
    new_sd = {}
    for k, v in state_dict.items():
        nk = k.replace("fc.1", "fc").replace("fc.0", "fc")
        new_sd[nk] = v
    backbone.load_state_dict(new_sd, strict=False)

    # wrap into extractor
    extractor = ResNet18Features(backbone).to(DEVICE).eval()
    return extractor, class_names, img_size, mean, std

@st.cache_resource
def load_xgb(pkl_path):
    """
    Load XGBoost model trained on CNN features.
    """
    if os.path.exists(pkl_path):
        import joblib
        return joblib.load(pkl_path)
    raise FileNotFoundError("XGBoost model (.json/.pkl) not found.")

# ===================== UI =====================
# Title and description
st.title("Identify 10 Pesticides and Dyes from Raman spectra by using MLRaman")
st.markdown(
    "MLRaman: Feature extractor: ResNet-18 • Classifier: XGBoost • "
    "Input: Raman spectrum image (PNG/JPG).  \n"
    "List of pesticides and dyes: CBZ, CPF, CR, CV, CYP, MP, R6G, RB, TBZ, TMTD"
)

# Load models
try:
    extractor, CLASS_NAMES, IMG_SIZE, MEAN, STD = load_feature_extractor_and_meta(CKPT_PTH)
    xgb_model = load_xgb(XGB_PKL)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


uploaded = st.file_uploader("Upload Raman spectrum image (PNG/JPG) to start identification", type=["png", "jpg", "jpeg"])
topk = st.sidebar.slider("Show Top-k Probabilities", min_value=1, max_value=10, value=5)

if uploaded is not None:
    # ---------------- LOAD FULL HD IMAGE (NO RESIZE) ----------------
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    original_img = Image.fromarray(rgb)

    # ----------- SHOW HD ORIGINAL IMAGE ------------
    st.image(
        original_img,
        caption="Processed Image",
        use_container_width=True     
    )

    # -------------------- CNN PREPROCESS --------------------
    tensor = preprocess_image_for_cnn(original_img, IMG_SIZE, MEAN, STD)

    # -------------------- CNN → FEATURES --------------------
    with torch.no_grad():
        feats = extractor(tensor.to(DEVICE))
    feats_np = feats.cpu().numpy()

    # -------------------- XGB PREDICT --------------------
    proba = xgb_model.predict_proba(feats_np)[0]
    pred_idx = int(np.argmax(proba))
    pred_name = CLASS_NAMES[pred_idx]

    st.success(f"🎯 Prediction: **{pred_name}**")

    order = np.argsort(proba)[::-1][:topk]

    st.subheader("Top-k Probabilities")
    st.table({
        "Class": [CLASS_NAMES[i] for i in order],
        "Probability": [f"{proba[i]:.3f}" for i in order]
    })

    # Note
    st.info(
        "Note: The classification model is based on features extracted from CNN (ResNet-18) "
        "and inferred by XGBoost. Please use images with clear bright/dark backgrounds, "
        "without cropping the signal region for best results."
    )

else:
    st.write("⬆️ Upload a Raman spectrum image to start identification.")

st.caption("Interdisciplinary Materials Group @ NTU (https://nguyen-group.github.io)")
