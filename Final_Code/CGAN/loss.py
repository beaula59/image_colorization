import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

class GANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla', real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()

    def get_labels(self, preds, target_is_real):
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds)

    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        vgg.eval()
        self.layers = nn.ModuleList([
            vgg[:4],    # relu1_1
            vgg[4:9],   # relu2_1
            vgg[9:16],  # relu3_1
            # Removed deeper layers (relu4_1, relu5_1) to make it faster
        ])
        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    # def normalize(self, x):
    #     return (x - self.mean) / self.std
    def normalize(self, x):
        # Convert mean and std to PyTorch tensors if they are numpy arrays
        # print(f"Device of x: {x.device}")
        x  = torch.tensor(x, dtype=torch.float32, device=x.device) if isinstance(x, np.ndarray) else x
        mean = torch.tensor(self.mean, dtype=torch.float32, device=x.device)
        std = torch.tensor(self.std, dtype=torch.float32, device=x.device)
        
        if x.shape[1] != 3:
            # transpose to 3 channels
            x = x.permute(0, 3, 1, 2)
            
        # print(f"x shape: {x.shape}, mean shape: {mean.shape}, std shape: {std.shape}")
        
        # # print type of x, mean, std
        # print(f"x type: {x.dtype}, mean type: {mean.dtype}, std type: {std.dtype}")
        # Now perform the operation on tensors, ensuring no type conflict
        return (x - mean) / std

    def forward(self, generated, real):
        # print(f"Generated shape: {generated.shape}, Real shape: {real.shape}")
        # Make sure input is 3-channel
        # if generated.shape[1] != 3:
        #     generated = generated.repeat(1, 3, 1, 1)
        #     real = real.repeat(1, 3, 1, 1)
        
        generated = self.normalize(generated)
        real  = torch.tensor(real, dtype=torch.float32, device=generated.device)
        # # print(f"Device of real: {real.device}")
        # Ensure real is on the same device as generated
        real = self.normalize(real)
        # generated = F.interpolate(generated, size=(128, 128), mode='bilinear', align_corners=False)
        # real = F.interpolate(real, size=(128, 128), mode='bilinear', align_corners=False)

        loss = 0.0

        with torch.no_grad():
            generated_feats = []
            real_feats = []
            for layer in self.layers:
                generated = layer(generated)
                real = layer(real)
                generated_feats.append(generated)
                real_feats.append(real)

        loss = 0.0
        for gen_feat, real_feat in zip(generated_feats, real_feats):
            loss += F.l1_loss(gen_feat, real_feat)

        return loss