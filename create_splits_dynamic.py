import os
import random
import math

# Path to your Images folder
images_folder = r"C:\Users\myasw\OneDrive\Desktop\Image caption generator\Filckr8k_Dataset\Images"
dataset_folder = r"C:\Users\myasw\OneDrive\Desktop\Image caption generator\Filckr8k_Dataset"

# List all jpg images
images = [f for f in os.listdir(images_folder) if f.lower().endswith('.jpg')]

# Shuffle images randomly
random.shuffle(images)

# Calculate split counts
total = len(images)
train_count = math.floor(total * 0.75)  # 75% for training
dev_count = math.floor(total * 0.125)   # 12.5% for dev/validation
test_count = total - train_count - dev_count  # remaining for test

# Split the images
train_images = images[:train_count]
dev_images = images[train_count:train_count + dev_count]
test_images = images[train_count + dev_count:]

# Save to text files
with open(os.path.join(dataset_folder, "Flickr_8k.trainImages.txt"), "w") as f:
    f.write("\n".join(train_images))

with open(os.path.join(dataset_folder, "Flickr_8k.devImages.txt"), "w") as f:
    f.write("\n".join(dev_images))

with open(os.path.join(dataset_folder, "Flickr_8k.testImages.txt"), "w") as f:
    f.write("\n".join(test_images))

print(f"Created splits: {len(train_images)} train, {len(dev_images)} dev, {len(test_images)} test")
