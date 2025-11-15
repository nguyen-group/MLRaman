import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from matplotlib.patches import Rectangle
import numpy as np

# ======= PLOTTING THEME =======
sns.set_theme(style="ticks", palette="muted", context="talk", font="Arial")

# Font style
plt.rcParams['font.family'] = 'Arial'

# ======= TRAINING CURVES PLOTTING =======
def plot_training_curves(train_accs, val_accs, train_losses, val_losses, output_dir, filename_prefix="training_curves"):
    plt.figure(figsize=(12,5))

    # ====== ACCURACY PLOT ======
    plt.subplot(1, 2, 1)
    sns.lineplot(x=range(1, len(train_accs)+1), y=train_accs, marker='o',
                 color="C3", markeredgecolor="none", label="Train Acc")
    sns.lineplot(x=range(1, len(val_accs)+1), y=val_accs, marker='s',
                 color="C0", markeredgecolor="none", label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xlim(0, len(train_accs) + 1)
    plt.legend(frameon=False)

    # ====== LOSS PLOT ======
    plt.subplot(1, 2, 2)
    sns.lineplot(x=range(1, len(train_losses)+1), y=train_losses, marker='o',
                 color="C3", markeredgecolor="none", label="Train Loss")
    sns.lineplot(x=range(1, len(val_losses)+1), y=val_losses, marker='s',
                 color="C0", markeredgecolor="none", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xlim(0, len(train_losses) + 1)
    plt.legend(frameon=False)

    # ====== SAVE PLOT ======
    plt.tight_layout()
    filename = f"{filename_prefix}_plot.pdf"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    print(f"✅ Plot saved to: {save_path}")

# ======= CONFUSION MATRIX PLOTTING =======
def plot_confusion_matrix(y_true, y_pred, class_names, out_dir="./output", 
                          model_name="model", cmap="Blues"):
    """
    Plot and save a styled confusion matrix with counts.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
    class_names : list of str
        Names of classes for axes labels.
    out_dir : str
        Directory where the confusion matrix images will be saved.
    model_name : str, optional
        Name for output file prefix (default: "model").
    cmap : str, optional
        Matplotlib colormap (default: "Blues").
    """

    os.makedirs(out_dir, exist_ok=True)
    n_classes = len(class_names)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    # ======= STYLE =======
    sns.set_theme(style="ticks", palette="muted", context="talk", font="Arial", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(7, 5.5))

    hm = sns.heatmap(
        cm,
        annot=True, fmt='d', cmap=cmap, cbar=True, square=True,
        xticklabels=class_names, yticklabels=class_names,
        linewidths=1.0, linecolor="white",
        annot_kws={"size": 18},
        ax=ax
    )

    # Adjust text color dynamically
    mx = cm.max() if cm.size else 0
    for t in hm.texts:
        try:
            v = int(t.get_text())
        except ValueError:
            continue
        t.set_color("white" if v > 0.6 * mx else "black")

    # Customize outline of heatmap
    for sp in ax.spines.values():
        sp.set_linewidth(2)
        sp.set_edgecolor("black")

    # Add a thicker border around the entire heatmap
    nrows, ncols = cm.shape
    ax.add_patch(Rectangle((0, 0), ncols, nrows,
                           fill=False, edgecolor="black", linewidth=2,
                           zorder=17, clip_on=False, transform=ax.transData))

    # Customize colorbar
    cbar = hm.collections[0].colorbar
    cbar.set_label("Count")

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    #ax.set_title(f"Confusion Matrix — {model_name}")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save outputs
    #cm_png = os.path.join(out_dir, f"confusion_matrix_{model_name}.png")
    cm_pdf = os.path.join(out_dir, f"confusion_matrix_{model_name}.pdf")
    #plt.savefig(cm_png, dpi=600, bbox_inches="tight")
    plt.savefig(cm_pdf, bbox_inches="tight")
    plt.show()

    print(f"✅ Saved Confusion Matrix -> {cm_pdf}")

# ======= ROC CURVES PLOTTING =======
def plot_roc_curves(
    fpr, tpr, roc_auc, class_names, out_dir,
    model_name="model",
    inset_zoom=False,
    zoom_limits=(0.00, 0.04, 0.9, 1.0)
):
    """
    Plot and save multi-class ROC curves with optional inset zoom view.

    Parameters
    ----------
    fpr : list of array-like
        False Positive Rates for each class.
    tpr : list of array-like
        True Positive Rates for each class.
    roc_auc : list or dict
        AUC values for each class.
    class_names : list of str
        Names of classes to label each curve.
    out_dir : str
        Directory where the ROC figure will be saved.
    model_name : str, optional
        Name for output file (default: "model").
    inset_zoom : bool, optional
        Whether to include an inset zoom subplot (default: False).
    zoom_limits : tuple, optional
        (x_min, x_max, y_min, y_max) limits for the inset zoom area.
    """

    os.makedirs(out_dir, exist_ok=True)
    sns.set_theme(style="ticks", palette="muted", context="talk", font="Arial")

    palette = [
        "#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1",
        "#76B7B2", "#EDC948", "#9C755F", "#BAB0AC", "#17BECF"
    ]

    fig, ax = plt.subplots(figsize=(6, 6))
    n_classes = len(class_names)

    # ---------- Main ROC curves ----------
    for i in range(n_classes):
        ax.plot(
            fpr[i], tpr[i],
            lw=2.0, color=palette[i % len(palette)],
            label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})"
        )

    # Baseline (random classifier)
    ax.plot([0, 1], [0, 1], ls="--", color="lightgray", lw=2)

    # Axis setup
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_aspect('equal')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=14, frameon=True)

    # ---------- Optional inset zoom ----------
    if inset_zoom:
        ins = ax.inset_axes([0.2, 0.6, 0.22, 0.35])
        ins.set_facecolor("white")

        for i in range(n_classes):
            ins.plot(fpr[i], tpr[i], lw=2, color=palette[i % len(palette)])
        ins.plot([0, 1], [0, 1], ls="--", color="gray", lw=0.9, alpha=0.45)

        x_min, x_max, y_min, y_max = zoom_limits
        ins.set_xlim(x_min, x_max)
        ins.set_ylim(y_min, y_max)
        ins.set_xticks(np.arange(x_min, x_max + 1e-6, 0.02))
        ins.set_yticks(np.arange(y_min, y_max + 1e-6, 0.05))
        ins.tick_params(labelsize=14, length=9)

    plt.tight_layout(rect=[0, 0, 0.92, 1])

    # ---------- Save ----------
    pdf_path = os.path.join(out_dir, f"roc_curve_{model_name}.pdf")
    #png_path = os.path.join(out_dir, f"roc_curve_{model_name}.png")
    plt.savefig(pdf_path, bbox_inches='tight')
    #plt.savefig(png_path, dpi=600, bbox_inches='tight')
    plt.show()

    print(f"✅ Saved ROC Curves -> {pdf_path}")

def plot_f1_scores(y_true, y_pred, class_names, out_dir="./output", model_name="model"):
    """
    Plot and save per-class F1 scores as a color-coded bar chart.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
    class_names : list of str
        Names of classes for x-axis labels.
    out_dir : str
        Directory where the plot will be saved.
    model_name : str, optional
        Name for the output file prefix (default: "model").
    """

    os.makedirs(out_dir, exist_ok=True)

    # ===== Compute metrics =====
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )

    # ===== Define colors (Tableau-10 palette) =====
    TABLEAU10 = [
        "#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1",
        "#76B7B2", "#EDC948", "#9C755F", "#BAB0AC", "#17BECF"
    ]
    color_map = dict(zip(class_names, TABLEAU10))
    bar_colors = [color_map[c] for c in class_names]

    # ===== Plot style =====
    sns.set_theme(style="ticks", palette="muted", context="talk", font="Arial")

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(class_names, f1, color=bar_colors, edgecolor="black", linewidth=1.1)

    # Add F1-score text labels above each bar
    #for bar, score in zip(bars, f1):
    #    ax.text(
    #        bar.get_x() + bar.get_width() / 2,
    #        bar.get_height() + 0.02,
    #        f"{score:.2f}",
    #        ha="center", va="bottom", fontsize=14
    #    )

    # ===== Axes setup =====
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Classes", labelpad=16)
    ax.set_ylabel("F1-score", labelpad=16)
    #ax.set_title(f"Per-Class F1-score — {model_name}", fontsize=26, pad=22)

    plt.tight_layout(rect=(0, 0, 1, 0.95))

    # ===== Save outputs =====
    #f1_png = os.path.join(out_dir, f"f1_scores_{model_name}.png")
    f1_pdf = os.path.join(out_dir, f"f1_scores_{model_name}.pdf")
    #plt.savefig(f1_png, dpi=800, bbox_inches="tight")
    plt.savefig(f1_pdf, bbox_inches="tight")
    plt.show()

    print(f"✅ Saved F1 Bar Plot -> {f1_pdf}")

# ======= DATASET DISTRIBUTION PLOTTING =======

def plot_dataset_distribution(out_dir="./output"):
    """
    Plot and save a pie chart showing dataset distribution across classes.
    """

    # Dataset distribution data
    class_names = ["CBZ","CR","TMTD","TBZ","R6G","RB","CV","MP","CYP","CPF"]
    sample_counts = [135, 128, 229, 239, 101, 100, 111, 104, 100, 100]

    plt.figure(figsize=(12,5))

    # Plot bar chart
    plt.subplot(1, 2, 1)

    sns.barplot(x=class_names, y=sample_counts)  # Draw bar plot

    # Customize plot
    #plt.title("Number of images per class", fontsize=14)
    plt.xlabel("Class Names")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Plot pie chart
    plt.subplot(1, 2, 2)
    plt.style.use('seaborn-v0_8-talk')

    colors = plt.cm.tab20c.colors[:10]

    def make_autopct(values):
        def my_autopct(pct):
            return '{p:.1f}%'.format(p=pct)
        return my_autopct

    wedges, texts, autotexts = plt.pie(
        sample_counts,
        labels=class_names,
        autopct=make_autopct(sample_counts),
        startangle=90,
        colors=colors,
        textprops={'fontsize': 16},
        wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'},
        pctdistance=0.8,
        labeldistance=1.08
    )

    for autotext in autotexts:
        autotext.set_fontsize(14)
        autotext.set_weight('bold')

    plt.axis('equal')
    plt.tight_layout()

    # ---------- Save ----------
    pdf_path = os.path.join(out_dir, f"class_distribution.pdf")
    #png_path = os.path.join(out_dir, f"class_distribution.png")
    plt.savefig(pdf_path, bbox_inches='tight')
    #plt.savefig(png_path, dpi=600, bbox_inches='tight')
    plt.show()

    print(f"✅ Saved Class Distribution -> {pdf_path}")