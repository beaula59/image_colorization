# From Gray to Vivid: Image Colorization with Deep Learning

This project explores advanced methods for image colorization using deep learning. We enhance existing approaches with curriculum learning, perceptual loss, and an ensemble of models, while proposing improved aesthetic evaluation metrics. Our experiments on ImageNet and COCO datasets show that these strategies improve color fidelity, perceptual quality, and overall robustness, with trade-offs tailored for specific goals like vividness or structural accuracy.

Sample images of our various approaches can be seen below:


![Image](https://github.com/user-attachments/assets/7ba257cc-7d91-42bd-bc22-8dd1e32f7845)


## Setup

We work with two datasets: **ImageNet** and **COCO**.

### 1. Download Datasets

- **COCO (Unlabeled 2017):**  
  Download from [https://cocodataset.org/#download](https://cocodataset.org/#download) → *unlabeled images (2017)*.

- **ImageNet:**  
  Navigate to the `ImageNet` folder and run the `download.sh` script:
  ```bash
  cd ImageNet
  ./download.sh
### 2. Prepare Test Image Directories

After downloading the datasets, run the following scripts to generate test image folders:

- For **COCO** (run from the project root directory):
  ```bash
  python create_test.py
  
- For **ImageNet**:

  ```bash
  cd ImageNet
  python create_test.py

### 3. Final Directory Structure

Your project directory should now look like this:

  ```plaintext
  Final_Code/
  ├── ImageNet/
  │   └── create_test.py
  ├── unlabeled2017/
  ├── test_COCO/
  ├── test_ImageNet/
  ├── ensemble/
  ├── cGAN/
  └── create_test.py
  ```

## Training the models:

Navigate to the `cGAN` directory

Baseline model:  
```bash
python3 pretrain.py
```

CL model:
```bash
python3 pretrain_CL.py
```

Perceptual model:
```bash
python3 pretrain_perceptual.py
```

## Testing
Navigate to the `cGAN` directory

```bash
python3 pretrain_test.py --checkpoint <path-to-ckpt> --save_path <path-to-folder-to-save-images>
```
