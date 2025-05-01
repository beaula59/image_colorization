from model_perceptual import PerceptualMainModel, build_res_unet
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from skimage.color import rgb2lab

import torch
from compute_metrics import compute_metrics
import glob
from skimage.color import lab2rgb
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# Set up argument parser
parser = argparse.ArgumentParser(description="Test the colorization model")
parser.add_argument('--checkpoint', type=str, required = True, help="Path to the model checkpoint")
parser.add_argument('--save_path', type=str, required = True, help="Path to save colorized images")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformation
SIZE = 256
class ColorizationDataset(Dataset):
    def __init__(self, paths):
        self.transforms = transforms.Compose([
            transforms.Resize((SIZE, SIZE), Image.BICUBIC),
            transforms.ToTensor()
        ])
        self.paths = paths
    
    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            img = self.transforms(img)
            img_np = np.array(img.permute(1, 2, 0))  # Convert to HWC format for rgb2lab
            img_lab = rgb2lab(img_np).astype("float32")
            img_lab = torch.tensor(img_lab).permute(2, 0, 1)  # Convert to CHW format
            L = img_lab[[0], ...] / 50. - 1.  # Normalize L
            ab = img_lab[[1, 2], ...] / 110.  # Normalize ab
            return {'L': L, 'ab': ab, 'path': self.paths[idx]}
        except Exception as e:
            print(f"Error loading image {self.paths[idx]}: {e}")
            return None  # Skip the image
    
    def __len__(self):
        return len(self.paths)

# load data:
coco_path = "../test_COCO"
imgnet_path = "../test_ImageNet"

coco_paths = sorted(glob.glob(coco_path + "/*.jpg"))
imgnet_paths = sorted(glob.glob(imgnet_path + "/**/*.JPEG", recursive=True))

# initialize model
net_G = build_res_unet(n_input=1, n_output=2, size=256)
model = PerceptualMainModel(net_G=net_G)
model.load_state_dict(torch.load(args.checkpoint, map_location=device))
model.to(device)
model.eval()

for test_paths in [imgnet_paths, coco_paths]:
    dataset = ColorizationDataset(test_paths)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True, drop_last=True)
    print(len(test_loader))

    colorized_results = []
    for batch in tqdm(test_loader, desc="Colorizing images"):
        if batch is None:
            continue  # Skip corrupted images

        L, ab, paths = batch['L'].to(device), batch['ab'].to(device), batch['path']
        
        with torch.no_grad():
            pred_ab = model.net_G(L)  # Output should be ab channels
        
        for i, img_path in enumerate(paths):
            L_img = (L[i].cpu().numpy() + 1.) * 50.  # De-normalize L
            pred_ab_img = pred_ab[i].cpu().numpy() * 110.  # De-normalize ab
            lab_img = np.concatenate((L_img, pred_ab_img), axis=0).transpose(1, 2, 0)
            colorized_rgb = (lab2rgb(lab_img) * 255).astype("uint8")
            
            colorized_results.append((img_path, colorized_rgb))
                
    save_path = f"{args.save_path}/{'coco' if test_paths == coco_paths else 'imagenet'}/"
    os.makedirs(save_path, exist_ok=True)
    
    for j, (img_path, img_rgb) in enumerate(colorized_results):
        if j % 100 == 0:  # Save every 10th image
            output_path = os.path.join(save_path, f"colorized_{j}.png")
            Image.fromarray(img_rgb).save(output_path)
    print(f"Saved every 100th image to {save_path}")

    # Compute metrics
    ssim_scores = []
    colorfulness_scores = []
    color_harmony_scores = []
    color_balance_scores = []
    pcqi_scores = []
    lpips_scores = []

    for orig_img, colorized_rgb in tqdm(colorized_results):
        orig_img = cv2.imread(orig_img)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        orig_img = np.array(orig_img)

        colorized_rgb = np.array(colorized_rgb)

        metrics = compute_metrics(orig_img, colorized_rgb)

        ssim_scores.append(metrics['SSIM'])
        colorfulness_scores.append(metrics['Colorfulness'])
        color_harmony_scores.append(metrics['Color Harmony'])
        color_balance_scores.append(metrics['Color Balance'])
        pcqi_scores.append(metrics['PCQI'])
        lpips_scores.append(metrics['LPIPS']) 

    # Calculate average scores
    avg_ssim = np.mean(ssim_scores)
    avg_colorfulness = np.mean(colorfulness_scores)
    avg_color_harmony = np.mean(color_harmony_scores)
    avg_color_balance = np.mean(color_balance_scores)
    avg_pcqi = np.mean(pcqi_scores)
    avg_lpips = np.mean(lpips_scores)

    # Calculate standard deviation scores
    std_ssim = np.std(ssim_scores)
    std_colorfulness = np.std(colorfulness_scores)
    std_color_harmony = np.std(color_harmony_scores)
    std_color_balance = np.std(color_balance_scores)
    std_pcqi = np.std(pcqi_scores)
    std_lpips = np.std(lpips_scores)

    if test_paths == coco_paths:
        print("COCO Dataset Metrics:")
    else:
        print("ImageNet Dataset Metrics:")

    print(f"Average SSIM: {avg_ssim:.3f} ± {std_ssim:.3f}")
    print(f"Average Colorfulness: {avg_colorfulness:.3f} ± {std_colorfulness:.3f}")
    print(f"Average Color Harmony: {avg_color_harmony:.3f} ± {std_color_harmony:.3f}")
    print(f"Average Color Balance: {avg_color_balance:.3f} ± {std_color_balance:.3f}")
    print(f"Average PCQI: {avg_pcqi:.3f} ± {std_pcqi:.3f}")
    print(f"Average LPIPS: {avg_lpips:.3f} ± {std_lpips:.3f}")
    
