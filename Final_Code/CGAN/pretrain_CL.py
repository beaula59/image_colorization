import glob
import numpy as np
from tqdm import tqdm
from skimage.color import rgb2lab, lab2rgb
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from torch import nn, optim
from torchvision.utils import make_grid

from dataset import ColorizationDataset, make_dataloaders
from model import MainModel, build_res_unet
from utils import AverageMeter, create_loss_meters, update_losses, log_results, visualize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load data:
coco_path = "../unlabeled2017"
imgnet_path = "../ImageNet"

coco_data = glob.glob(coco_path + "/*.jpg")
imgnet_data = glob.glob(imgnet_path + "/**/*.JPEG", recursive=True)

np.random.seed(123)

dataset_size = 104229
paths_subset = np.random.choice(imgnet_data, dataset_size, replace=False) # choosing 1000 images randomly
rand_idxs = np.random.permutation(dataset_size)
train_idxs = rand_idxs[:int(dataset_size * 0.8)] # choosing the first 8000 as training set
val_idxs = rand_idxs[int(dataset_size * 0.8) : ] # choosing last 2000 as validation set
train_paths = paths_subset[train_idxs]
val_paths = paths_subset[val_idxs]
print(len(train_paths), len(val_paths))

# create dataset
train_dataset = ColorizationDataset(train_paths, split = 'train')
val_dataset = ColorizationDataset(val_paths, split = 'val')

# create dataloaders
train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
# train_dl = make_dataloaders(paths=train_paths, split='train')
# val_dl = make_dataloaders(paths=val_paths, split='val')

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

def train_model(model, train_dl, epochs, display_every=1000):
    # data = next(iter(val_dl)) # getting a batch for visualizing the model output after fixed intrvals
    train_set_size = int(dataset_size * 0.8) // 2

    for e in range(epochs):
        loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to
        i = 0                                  # log the losses of the complete network

        # curriculum learning
        q = np.sin((np.pi * e) / (2 * epochs)) # curriculum learning parameter

        # creating dataloader with q samples of coco dataset and (1-q) samples of imgnet dataset
        coco_size = int(train_set_size * q)
        imgnet_size = train_set_size - coco_size
        coco_size += train_set_size
        coco_paths = np.random.choice(coco_data, coco_size, replace=False)
        imgnet_paths = np.random.choice(imgnet_data, imgnet_size, replace=False)

        train_paths = np.concatenate((coco_paths, imgnet_paths), axis=0)
        train_dataset.update_paths(train_paths) # updating the dataset with new paths
        # train_dl = make_dataloaders(paths=train_paths, split='train')
        print(f"Epoch {e+1}/{epochs}: Training with {coco_size} coco images and {imgnet_size} imgnet images")
        
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

        if e % 10 == 0:
            # save model
            model_path = f"checkpoints/pretrain/CL/cGAN_CL_model_epoch_{e}.pth"
            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'loss_meter_dict': loss_meter_dict
            }, model_path)

# pretraining the generator
# net_G = build_res_unet(n_input=1, n_output=2, size=256)
# opt = optim.Adam(net_G.parameters(), lr=1e-4)
# criterion = nn.L1Loss()
# num_epochs_pretrain = 30
# pretrain_generator(net_G, train_dl, opt, criterion, num_epochs_pretrain)
# torch.save(net_G.state_dict(), "checkpoints/pretrain/CL/res18-unet.pt")

net_G = build_res_unet(n_input=1, n_output=2, size=256)
net_G.load_state_dict(torch.load("/u/student/2021/cs21btech11002/CV_Project/cGAN/res18-unet.pt", map_location=device))
model = MainModel(net_G=net_G)
num_epochs = 70
train_model(model, train_dl, num_epochs)
torch.save(model.state_dict(), "checkpoints/pretrain/CL/cGAN-unet_challengeCL.pt")
