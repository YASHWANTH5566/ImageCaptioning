import os

images_folder = r"C:\Users\myasw\OneDrive\Desktop\Image caption generator\Filckr8k_Dataset\Images"

images = [f for f in os.listdir(images_folder) if f.lower().endswith('.jpg')]
print("Number of images:", len(images))