from glob import glob
import os

os.makedirs('../test_ImageNet', exist_ok=True)
tot_imgs = len(glob('./**/*.JPEG', recursive=True))
print(f'Total images: {tot_imgs}')

desired_test_imgs = 10000
test_folder = './test_ImageNet'

for folder in os.listdir('./'):
    if os.path.isdir(folder):
        images = glob(f'{folder}/*.JPEG')
        num_imgs = len(images)

        # Calculate the proportional number of test images for this folder
        ratio = num_imgs / tot_imgs
        num_test_imgs = int(desired_test_imgs * ratio)
        print(f'Folder: {folder}, Number of images: {num_imgs}, Ratio: {ratio:.4f}, Number of test images: {num_test_imgs}')

        # Select the first `num_test_imgs` images for testing
        test_images = images[:num_test_imgs]
        for img in test_images:
            os.rename(img, os.path.join(test_folder, os.path.basename(img)))
        
