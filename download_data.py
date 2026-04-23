import os

print("Downloading dataset from Kaggle...")

os.system("kaggle datasets download -d sshikamaru/car-object-detection")
os.system("unzip car-object-detection.zip -d data/")

print("Dataset ready in /data folder")