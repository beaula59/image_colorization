from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb


SIZE = 256
class ColorizationDataset(Dataset):
    def __init__(self, paths, split='train'):
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((SIZE, SIZE),  Image.BICUBIC),
                transforms.RandomHorizontalFlip(), # A little data augmentation!
            ])
        elif split == 'val':
            self.transforms = transforms.Resize((SIZE, SIZE),  Image.BICUBIC)
        elif split == 'test':
            self.transforms = transforms.Compose([
                transforms.Resize((SIZE, SIZE), Image.BICUBIC),
                transforms.ToTensor()
            ])

        self.split = split
        self.size = SIZE
        self.paths = paths

    def __getitem__(self, idx):

        # if (self.paths[idx].endswith('.jpg')) and self.split == 'train':
        #     print("coco")

        while (True):
            try:
                img = Image.open(self.paths[idx]).convert("RGB")
                break
            except OSError as e:
                print(f"Skipping corrupted image: {self.paths[idx]} ({e})")
                idx = (idx + 1) % len(self.paths)

        # img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)

        if self.split == 'test':
            img = np.array(img.permute(1, 2, 0))

        img_lab = rgb2lab(img).astype("float32") # Converting RGB to L*a*b
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1. # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110. # Between -1 and 1

        if self.split == 'test':
            return {'L': L, 'ab': ab, 'path': self.paths[idx]}
        return {'L': L, 'ab': ab}

    def __len__(self):
        return len(self.paths)

    def update_paths(self, paths):
        self.paths = paths

def make_dataloaders(batch_size=64, n_workers=4, pin_memory=True, **kwargs): # A handy function to make our dataloaders
    dataset = ColorizationDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers,
                            pin_memory=pin_memory)
    return dataloader