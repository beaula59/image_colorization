import os
import random
import shutil

# Source and destination directories
source_dir = './unlabeled2017'
destination_dir = './test_COCO'

# Create destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Get a list of all files in the source directory
all_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

# Select 10,000 random files
selected_files = random.sample(all_files, min(10000, len(all_files)))

# Move selected files to the destination directory
for file_name in selected_files:
    shutil.move(os.path.join(source_dir, file_name), os.path.join(destination_dir, file_name))

print(f"Selected {len(selected_files)} files and copied them to {destination_dir}.")