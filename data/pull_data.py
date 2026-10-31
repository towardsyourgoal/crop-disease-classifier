import kagglehub
import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


image_ext=('.jpg', '.jpeg', '.png')

# Download latest version

def find_class_root(path):
    """Return the folder that has the most subfolders containing images directly."""
    best, best_n = None, 0
    for dirpath, dirnames, _ in os.walk(path):          # top-down, so the outer folder wins ties
        # only using color folder
        dirnames[:] = [d for d in dirnames if d.lower() not in ('grayscale', 'segmented')]
        n = sum(any(f.lower().endswith(image_ext) for f in os.listdir(os.path.join(dirpath, d)))
                for d in dirnames)
        if n > best_n:
            best, best_n = dirpath, n
    return best

def get_data():
    # path = kagglehub.dataset_download("emmarex/plantdisease") # 20K images, 15 classes
    # changed to this as mentioned in the task  # 
    path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")
    print("Path dataset:", path)
    root = find_class_root(path)
    print("folders are found in:", root)
        
    dataset_transformed=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()]
    )
    full_ds=ImageFolder(root=root, 
                        transform=dataset_transformed)

    classes=full_ds.classes

    return full_ds, classes

