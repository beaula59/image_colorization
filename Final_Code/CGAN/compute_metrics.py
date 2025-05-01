import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import gaussian_filter
import lpips
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy import linalg

# Load LPIPS model once (global for reuse)
_lpips_model = lpips.LPIPS(net='vgg')
_lpips_model = _lpips_model.cuda() if torch.cuda.is_available() else _lpips_model

def compute_ssim(original, colorized):
    """Computes SSIM between two color images."""
    SIZE = 256  # Ensure both images are 256x256
    original_resized = cv2.resize(original, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
    colorized_resized = cv2.resize(colorized, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
    if original_resized.ndim == 2:
        original_resized = cv2.cvtColor(original_resized, cv2.COLOR_GRAY2RGB)
    # Compute SSIM for RGB images
    return ssim(original_resized, colorized_resized, data_range=255, channel_axis=2)

def compute_colorfulness(image):
    """Computes the Colorfulness Index (CI)."""
    (R, G, B) = cv2.split(image)
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    std_rg, std_yb = np.std(rg), np.std(yb)
    mean_rg, mean_yb = np.mean(rg), np.mean(yb)
    return np.sqrt(std_rg**2 + std_yb**2) + (0.3 * np.sqrt(mean_rg**2 + mean_yb**2))

def compute_color_harmony(image):
    """Computes a color harmony score based on hue variance."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].flatten()
    hue_variance = np.var(hue)
    return 1 / (1 + hue_variance)

def compute_color_distribution_balance(image, num_bins=32):
    """Computes color distribution balance using histogram entropy."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [num_bins], [0, 180])
    h_hist = h_hist / np.sum(h_hist)  # Normalize histogram
    return 1 - np.sum(h_hist * np.log(h_hist + 1e-6)) / np.log(num_bins)

# --- Define PCQI Function ---
def compute_pcqi(original, colorized, sigma=1.5):
    """Computes PCQI (Perception-based Contrast Quality Index) for color images."""

    SIZE = 256  # Ensure both images are 256x256
    original = cv2.resize(original, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
    colorized = cv2.resize(colorized, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)

    # Convert images to YIQ color space
    original_yiq = cv2.cvtColor(original, cv2.COLOR_RGB2YCrCb)
    colorized_yiq = cv2.cvtColor(colorized, cv2.COLOR_RGB2YCrCb)

    # Extract luminance (Y) and chrominance (I, Q) channels
    Y_orig, I_orig, Q_orig = cv2.split(original_yiq)
    Y_col, I_col, Q_col = cv2.split(colorized_yiq)

    # --- Compute Structural Similarity Index (SSIM) for Luminance ---
    luminance_similarity = ssim(Y_orig, Y_col, data_range=Y_orig.max() - Y_orig.min())

    # --- Compute Chrominance Similarity ---
    def chroma_similarity(channel_orig, channel_col):
        mu_x = gaussian_filter(channel_orig, sigma)
        mu_y = gaussian_filter(channel_col, sigma)
        sigma_x = np.sqrt(gaussian_filter(channel_orig**2, sigma) - mu_x**2)
        sigma_y = np.sqrt(gaussian_filter(channel_col**2, sigma) - mu_y**2)
        contrast_measure = (2 * sigma_x * sigma_y) / (sigma_x**2 + sigma_y**2 + 1e-6)
        structure_measure = (2 * mu_x * mu_y) / (mu_x**2 + mu_y**2 + 1e-6)
        return np.mean(contrast_measure * structure_measure)

    I_similarity = chroma_similarity(I_orig, I_col)
    Q_similarity = chroma_similarity(Q_orig, Q_col)
    chrominance_similarity = (I_similarity + Q_similarity) / 2

    # --- Final PCQI Score ---
    pcqi_score = 0.5 * luminance_similarity + 0.5 * chrominance_similarity
    return pcqi_score

def compute_lpips(img1, img2):
    """
    Computes LPIPS (Learned Perceptual Image Patch Similarity) between two RGB images.
    Images must be in RGB format, shape (H, W, 3), and dtype uint8 or float32 in [0, 255].

    Returns a float LPIPS score.
    """
    SIZE = 256  # Resize both images
    img1_resized = cv2.resize(img1, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
    img2_resized = cv2.resize(img2, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)

    # Convert to float32 and normalize to [-1, 1]
    def preprocess(img):
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) * 2.0  # [0,1] -> [-1,1]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return img_tensor.cuda() if torch.cuda.is_available() else img_tensor

    img1_tensor = preprocess(img1_resized)
    img2_tensor = preprocess(img2_resized)

    # Compute LPIPS distance
    with torch.no_grad():
        dist = _lpips_model(img1_tensor, img2_tensor)
    return dist.item()


def compute_fid(real_img, fake_img, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # --- Image transform
    def get_transform(image_size=299):
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
    
    # --- Feature extraction for one image
    def extract_features(image_tensor, model):
        model.eval()
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(0).to(device)
            features = model(image_tensor)
        return features.squeeze(0).cpu().numpy()

    # --- FID formula
    def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
        covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = np.sum((mu1 - mu2) ** 2) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid

    # --- Main logic
    transform = get_transform()
    # Convert NumPy array (H x W x C) to PIL Image
    if isinstance(real_img, np.ndarray):
        #print("Converting real_img to PIL Image")
        real_img = Image.fromarray(real_img.astype('uint8'))

    if isinstance(fake_img, np.ndarray):
        #print("Converting fake_img to PIL Image")
        fake_img = Image.fromarray(fake_img.astype('uint8'))
    real_tensor = transform(real_img).to(device)
    fake_tensor = transform(fake_img).to(device)

    weights = Inception_V3_Weights.DEFAULT
    inception = inception_v3(weights=weights, aux_logits=True).to(device)
    inception.fc = torch.nn.Identity()
    inception.eval()

    real_feat = extract_features(real_tensor, inception)
    fake_feat = extract_features(fake_tensor, inception)

    real_feats = real_feat.reshape(1, -1)
    fake_feats = fake_feat.reshape(1, -1)

    mu1, sigma1 = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False) + np.eye(real_feats.shape[1]) * 1e-6
    mu2, sigma2 = fake_feats.mean(axis=0), np.cov(fake_feats, rowvar=False) + np.eye(fake_feats.shape[1]) * 1e-6

    fid_score = calculate_fid(mu1, sigma1, mu2, sigma2)
    return fid_score


# --- Define Function to Compute All Metrics ---
def compute_metrics(original, colorized):
    """Computes various metrics for colorization quality."""
    ssim_value = compute_ssim(original, colorized)
    colorfulness_value = compute_colorfulness(colorized)
    color_harmony_value = compute_color_harmony(colorized)
    color_balance_value = compute_color_distribution_balance(colorized)
    pcqi_value = compute_pcqi(original, colorized)
    lpips_value = compute_lpips(original, colorized)
    # fid_score = compute_fid(original, colorized)

    return {
        "SSIM": ssim_value,
        "Colorfulness": colorfulness_value,
        "Color Harmony": color_harmony_value,
        "Color Balance": color_balance_value,
        "PCQI": pcqi_value,
        "LPIPS": lpips_value
        # "FID Score": fid_score
    }