# MLRaman
Rapid Machine Learning–Driven Detection of Pesticides and Dyes Using Raman Spectroscopy

![MLRaman-model](https://github.com/user-attachments/assets/1925967b-17ea-4952-8733-588869d2938b)

# Requirement
MLRaman requires the packages as follows: 
- `torch`: an open-source machine learning library with strong GPU acceleration.
- `jupyterlab`: a web-based interactive development environment for notebooks, code, and data.
- `matplotlib`: a comprehensive library for creating static, animated, and interactive visualizations in Python.
- `seaborn`: a Python data visualization library based on matplotlib.
- `scikit-learn`: a set of python modules for machine learning and data mining
- `python-docx`: a Python library for creating and updating Microsoft Word (.docx) files
- `imagehash`: a Python library for image hashing.
- `opencv-python`: a computer vision library, containing over 2500 algorithms, and is free for commercial use.
- `xgboost`: a machine learning library based on the gradient boosted decision trees algorithm.
- `openTSNE`: a modular Python implementation of t-Distributed Stochasitc Neighbor Embedding (t-SNE).
- `umap-learn`: a dimension reduction technique that can be used for the UMAP method.
- `streamlit`: a Python framework for data scientists and AI/ML engineers to deliver interactive data apps.

Example to install requirements with conda for CPU & GPU:
```md
$ conda create -n torch python=3.9
$ conda activate torch
$ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
$ pip3 install jupyterlab matplotlib seaborn scikit-learn python-docx imagehash opencv-python xgboost openTSNE umap-learn streamlit
```

# Directory Description

```md
MLRaman
├── data
│   └── docx: a dataset of Raman spectra of 10 pesticides and dyes, which humans collect from the literature.
│   └── test: the unseen experimental Raman spectra of CV and R6B for testing the app.
├── utils
│   ├── utils_data.py: define the function related to the dataset.
│   ├── utils_model.py: define the function related to the model.
│   └── utils_plot.py: define the function related to plotting results.
├── model
│   ├── resnet18_10cls_20251115.pth: a trained GNN model.
├── output
│   ├── xgb_on_cnn.pkl: a trained XGBoost model with GNN features and PCA.
├── MLRaman.ipynb: Main MLRaman code
└── app_xgb_streamlit.py: user-friendly Streamlit application for real-time prediction.
```
# How to run
Step 1: Download the GNNOpt package:

    git clone https://github.com/nguyen-group/GNNOpt.git

Step 2: Go to the source code in the GNNOpt directory to run the program:

    cd MLRaman
    jupyter-lab MLRaman.ipynb

Step 3: For Streamlit application:

    streamlit run app_xgb_streamlit.py

Note: app_xgb_streamlit.py will load resnet18_10cls_20251115.pth and xgb_on_cnn.pkl, which are stored in model and output directories, respectively.

# References and citing
The detailed MLRaman is described in our pre-print:
> TBA
