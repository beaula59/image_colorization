import glob
import numpy as np
from tqdm import tqdm
from skimage.color import rgb2lab, lab2rgb
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
from torch import nn, optim
from torchvision.utils import make_grid

from dataset import ColorizationDataset, make_dataloaders
from model import MainModel, build_res_unet
from utils import AverageMeter, create_loss_meters, update_losses, log_results, visualize
import argparse

parser = argparse.ArgumentParser(description="Train the colorization model")
parser.add_argument('--data', default = 'mixed', choices = ['coco', 'mixed'])
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load data:
coco_path = "../unlabeled2017"
imgnet_path = "../ImageNet"
if (args.data == 'mixed'):
    paths = glob.glob(coco_path + "/*.jpg") + glob.glob(imgnet_path + "/**/*.JPEG", recursive=True)
else:
    paths = glob.glob(coco_path + "/*.jpg")
np.random.seed(123)

dataset_size = 104229
paths_subset = np.random.choice(paths, dataset_size, replace=False) # choosing 1000 images randomly
rand_idxs = np.random.permutation(dataset_size)
train_idxs = rand_idxs[:int(dataset_size * 0.8)] # choosing the first 8000 as training set
val_idxs = rand_idxs[int(dataset_size * 0.8) : ] # choosing last 2000 as validation set
train_paths = paths_subset[train_idxs]
val_paths = paths_subset[val_idxs]
print(len(train_paths), len(val_paths))

# create dataloader
train_dl = make_dataloaders(paths=train_paths, split='train')

jpg_count = sum(1 for path in train_paths if path.lower().endswith('.jpg'))
jpeg_count = sum(1 for path in train_paths if path.lower().endswith('.jpeg'))
total_count = len(train_paths)

jpg_percent = (jpg_count / total_count) * 100
jpeg_percent = (jpeg_count / total_count) * 100

print(f"Percentage of COCO in train: {jpg_percent:.2f}%")
print(f"Percentage of ImageNet in train: {jpeg_percent:.2f}%")

val_dl = make_dataloaders(paths=val_paths, split='val')

data = next(iter(train_dl))
Ls, abs_ = data['L'], data['ab']
print(Ls.shape, abs_.shape)
print(len(train_dl), len(val_dl))

# initialize model
# discriminator = PatchDiscriminator(3)
# batch_size = 64
# dummy_input = torch.randn(64, 3, 256, 256) # batch_size, channels, size, size
# out = discriminator(dummy_input)

def pretrain_generator(net_G, train_dl, opt, criterion, epochs):
    for e in range(epochs):
        loss_meter = AverageMeter()
        for data in tqdm(train_dl):
            L, ab = data['L'].to(device), data['ab'].to(device)
            preds = net_G(L)
            loss = criterion(preds, ab)
            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_meter.update(loss.item(), L.size(0))

        print(f"Epoch {e + 1}/{epochs}")
        print(f"L1 Loss: {loss_meter.avg:.5f}")
        print(f"Perceptual Loss: {loss_meter.avg:.5f}")

def train_model(model, train_dl, epochs, display_every=1000):
    data = next(iter(val_dl)) # getting a batch for visualizing the model output after fixed intrvals
    for e in range(epochs):
        loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to
        i = 0                                  # log the losses of the complete network
        for data in tqdm(train_dl):
            model.setup_input(data)
            model.optimize()
            update_losses(model, loss_meter_dict, count=data['L'].size(0)) # function updating the log objects
            i += 1
            if i % display_every == 0:
                print(f"\nEpoch {e+1}/{epochs}")
                print(f"Iteration {i}/{len(train_dl)}")
                log_results(loss_meter_dict) # function to print out the losses
                # visualize(model, data, save=False) # function displaying the model's outputs

# pretraining the generator
# net_G = build_res_unet(n_input=1, n_output=2, size=256)
# opt = optim.Adam(net_G.parameters(), lr=1e-4)
# criterion = nn.L1Loss()
# num_epochs_pretrain = 30
# pretrain_generator(net_G, train_dl, opt, criterion, num_epochs_pretrain)
# torch.save(net_G.state_dict(), "checkpoints/pretrain/baseline/res18-unet.pt")

net_G = build_res_unet(n_input=1, n_output=2, size=256)
net_G.load_state_dict(torch.load("/u/student/2021/cs21btech11002/CV_Project/cGAN/res18-unet.pt", map_location=device))
model = MainModel(net_G=net_G)
num_epochs = 70
train_model(model, train_dl, num_epochs)
torch.save(model.state_dict(), "checkpoints/pretrain/baseline_coco/cGAN-unet_cocopretrain.pt")
