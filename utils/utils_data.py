import os
from docx import Document
from io import BytesIO
import shutil, hashlib, random
import imagehash
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve

# ======= PLOTTING THEME =======
sns.set_theme(style="ticks", palette="muted", context="talk", font="Arial")

# Font style
plt.rcParams['font.family'] = 'Arial'

# Image hash function
def image_content_hash(image):
    return str(imagehash.phash(image.convert("RGB")))

# Extract images from a document part
def extract_images_from_part(part, label_dir, label, hash_set):
    count = 0
    for rel in part.rels.values():
        target_ref = getattr(rel, "target_ref", "")
        if "image" in target_ref:
            image_data = rel.target_part.blob
            try:
                image = Image.open(BytesIO(image_data))
                if image.width >= 50 and image.height >= 50:
                    img_hash = image_content_hash(image)
                    if img_hash not in hash_set:
                        ext = image.format.lower() if image.format else "png"
                        img_name = f"{label}_{count}.{ext}"
                        image.save(os.path.join(label_dir, img_name))
                        hash_set.add(img_hash)
                        count += 1
            except OSError:
                print(f"⚠️ Error reading image at {label}")
    return count

# Get all images from docx in a stable order
def get_all_images_sorted(doc):
    rels = doc.part._rels
    items = []

    for key in rels:
        rel = rels[key]
        if "image" in rel.target_ref:
            items.append((rel.rId, rel))

    # ===== Sort for rId for reproducible =====
    items.sort(key=lambda x: int(x[0][3:]))  # rId123 → 123

    # return list of image blobs in stable order
    return [rel.target_part.blob for _, rel in items]


def copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def enhance_image(in_path, out_path, upscale=False):
    """
    Strong image enhancement: denoise, sharpen, and boost contrast.
    """
    try:
        img = Image.open(in_path).convert("RGB")

        # Step 1: optional upscale for small images
        if upscale:
            w, h = img.size
            if min(w, h) < 256:  # threshold for small images
                img = img.resize((w*2, h*2), Image.LANCZOS)

        # Step 2: light denoise / smooth edges
        img = img.filter(ImageFilter.MedianFilter(size=3))

        # Step 3: boost brightness & contrast
        img = ImageEnhance.Brightness(img).enhance(1.00)  # +00%
        img = ImageEnhance.Contrast(img).enhance(1.30)   # +30%

        # Step 4: sharpen strongly
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        img = ImageEnhance.Sharpness(img).enhance(1.50) # +50%

        # Step 5: save with high quality
        img.save(out_path, quality=100, subsampling=0, optimize=True)
    except Exception as e:
        print(f"⚠️ {in_path}: {e}")

def enhance_image_cv(in_path, out_path, upscale=False):
    img = cv2.imread(in_path)
    if img is None:
        print(f"⚠️ Cannot open {in_path}")
        return

    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Step 1: optional upscale
    if upscale and min(img.shape[:2]) < 256:
        img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2), interpolation=cv2.INTER_CUBIC)

    # Step 2: denoise
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    # Step 3: enhance local contrast (CLAHE)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl,a,b))
    img = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    # Step 4: sharpen edges (kernel)
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    img = cv2.filter2D(img, -1, kernel)

    # Step 5: optional mild gamma correction
    gamma = 1.1
    img = np.clip((img / 255.0) ** (1.0/gamma) * 255.0, 0, 255).astype(np.uint8)

    # Save high-quality image
    Image.fromarray(img).save(out_path, quality=98, subsampling=0, optimize=True)

# Copy or move file to destination, avoiding name conflicts
def copy_or_move(src, dst):
    dst_dir = os.path.dirname(dst)
    base, ext = os.path.splitext(os.path.basename(dst))
    dst_path = os.path.join(dst_dir, base + ext)
    i = 1
    while os.path.exists(dst_path):
        dst_path = os.path.join(dst_dir, f"{base}_{i}{ext}")
        i += 1
    if MOVE_FILES:
        shutil.move(src, dst_path)
    else:
        shutil.copy2(src, dst_path)
# ========================================

# Process and plot spectrums
def process_and_plot_spectrums(image_path, out_dir="./output"):
    '''Process Raman spectrum image: baseline correction (Asymmetric Least Squares) + smoothing (Savitzky-Golay).
    Plot comparison of raw and processed spectrum.
    Args:
        image_path (str): Path to the Raman spectrum image file.
    '''

    # --- Load real spectral image and convert to grayscale 1D profile ---
    def load_spectrum_from_image(image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read {image_path}")

        spectrum = 255 - img.mean(axis=0) # Invert and average over height
        return spectrum


    # --- Baseline Correction using Asymmetric Least Squares ---
    def baseline_als(y, lam=1e5, p=0.01, niter=10):
        L = len(y)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2)).toarray()
        D = sparse.csc_matrix(D)
        w = np.ones(L)
        for _ in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1-p) * (y < z)
        return z
    
    # --- Main processing ---
    raw_spectrum = load_spectrum_from_image(image_path)

    # Baseline correction
    baseline = baseline_als(raw_spectrum)
    corrected = raw_spectrum - baseline

    # Smoothing
    smoothed = savgol_filter(corrected, window_length=21, polyorder=3)

    # --- Plot comparison ---
    x = np.linspace(100, 2000, len(raw_spectrum))  # Raman shift scale (customize if needed)

    # ======= PLOTTING =======
    sns.set_theme(style="ticks", palette="muted", context="talk", font="Arial")

    plt.figure(figsize=(6.5,5))

    sns.lineplot(x=x, y=raw_spectrum, label='Raw Spectrum', color='gray')
    sns.lineplot(x=x, y=baseline, label='Estimated Baseline', linestyle='--', color='orange')
    sns.lineplot(x=x, y=smoothed, label='ALS + SG', color='blue')
    plt.xlabel('Raman Shift (cm$^{-1}$)')
    plt.ylabel('Raman Intensity (a.u.)')
    # plt.title('Comparison of Raw and Preprocessed Raman Spectrum (Real Sample)')
    # Legend without frame
    plt.legend(frameon=False)
    # plt.grid(True)
    plt.tight_layout()

    # ---------- Save ----------
    pdf_path = os.path.join(out_dir, f"preprocessing_comparison_sample.pdf")
    #png_path = os.path.join(out_dir, f"preprocessing_comparison_sample.png")
    plt.savefig(pdf_path, bbox_inches='tight')
    #plt.savefig(png_path, dpi=600, bbox_inches='tight')
    plt.show()

    print(f"✅ Saved Class Distribution -> {pdf_path}")