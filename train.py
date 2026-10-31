import os
import torch
import torch.nn as nn
import copy
import time
import torchvision
import hashlib
import pandas as pd
import random

from data.pull_data import get_data
from PIL import Image
from torchvision import transforms as T
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


# Pull data from source
full_dataset, class_labels=get_data()
raw_df=pd.DataFrame(full_dataset.samples, columns=["path", "label"])
raw_df['class']=raw_df['label'].map(lambda ix:class_labels[ix])
raw_df["hash_key"]=[hashlib.md5(open(p,'rb').read()).hexdigest() for p in raw_df.path]
print(f"duplicate images {raw_df.hash_key.duplicated().sum()}")


# Preprocess data

# 1. remove duplicates 
# results - we have duplicate images inside the same class, so we can keep only one drop the extra

dup=raw_df[raw_df.duplicated("hash_key",keep=False)]
print(f'{len(dup)} files in {dup.hash_key.nunique()} duplicate groups')
conflict=dup.groupby('hash_key')['class'].nunique()
print('groups with conflicting classes:', (conflict > 1).sum())

before = len(raw_df)
raw_df = raw_df.drop_duplicates('hash_key', keep='first').reset_index(drop=True)
print(f'removed {before - len(raw_df)} duplicate files')


# 2. selecting ONLY 12 random classes for training

random.seed(42)

all_classes = sorted(raw_df['class'].unique())
selected = sorted(random.sample(all_classes, 12))
raw_df2 = raw_df[raw_df['class'].isin(selected)].reset_index(drop=True)
raw_df2['label'] = raw_df2['class'].map({c: i for i, c in enumerate(selected)})

classes=sorted(selected)

# 3. Stratified split (~70/20/10). before augmentation

train_df, temp = train_test_split(raw_df2, test_size=0.30, stratify=raw_df2.label, random_state=42)
val_df, test_df = train_test_split(temp, test_size=0.35, stratify=temp.label, random_state=42)

# 4. Augmentation

# resizing, horizontal flipping, vertical flipping, rotating, scaling - For training data
# resizing, scaling - For validation and test data

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] # standard ImageNet values 
aug_train_df = T.Compose(
    [T.RandomResizedCrop(224, scale=(0.7, 1.0)), 
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(30), 
    T.ToTensor(),
    T.Normalize(mean, std)])
aug_val_df = T.Compose([
    T.Resize((224, 224)), 
    T.ToTensor(), 
    T.Normalize(mean, std)])


# Handle imbalanced classes  
# we will perform this during training

# n = train_df.label.value_counts().sort_index().values
# w = (len(train_df) / (len(classes) * n)) ** 0.5
# criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))


# Build dataset and dataloader

class plantdataset(Dataset):
  def __init__(self, df, aug_df):
    self.paths=df["path"].tolist()
    self.labels=df["label"].tolist()
    self.transform=aug_df

  def __len__(self):
    return len(self.paths)

  def __getitem__(self, i):
    img=Image.open(self.paths[i]).convert('RGB')
    return self.transform(img), self.labels[i]

BATCH_SIZE=32
NUM_WORKERS=min(4, os.cpu_count() or 2)
pin=torch.cuda.is_available()

train_loader=DataLoader(plantdataset(train_df, aug_train_df),
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=NUM_WORKERS,
                        pin_memory=pin,
                        drop_last=True)

val_loader=DataLoader(plantdataset(val_df, aug_val_df),
                      batch_size=BATCH_SIZE,
                      shuffle=False,
                      num_workers=NUM_WORKERS,
                      pin_memory=pin,
                      drop_last=False)

test_loader=DataLoader(plantdataset(test_df, aug_val_df),
                       batch_size=BATCH_SIZE,
                       shuffle=False,
                       num_workers=NUM_WORKERS,
                       pin_memory=pin,
                       drop_last=False)

# val/test are not shuffled and keep every image, so prediction i matches row i of val_df/test_df
print(len(train_loader), len(val_loader), len(test_loader) , " batches")

# device setup , if cuda if running through a GPU
device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
K=len(selected)

use_amp = device.type == 'cuda'

class SmallCNN(nn.Module):
    def __init__(self, k):
        super().__init__()
        def block(i, o):
          return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False),
                               nn.BatchNorm2d(o),
                               nn.ReLU(inplace=True),
                               nn.MaxPool2d(2)
                               )
        self.features = nn.Sequential(block(3, 32),
                                      block(32, 64),
                                      block(64, 128),
                                      block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Flatten(),
                                  nn.Dropout(0.3),
                                  nn.Linear(256, k))
    def forward(self, x):
      return self.head(self.features(x))

def build_model(name):
    if name == 'small_cnn':
        return SmallCNN(K)
    if name == 'mobilenet_v3_small':
        m = torchvision.models.mobilenet_v3_small(
            weights=torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
            )
        m.classifier[3] = nn.Linear(
            m.classifier[3].in_features, K)
        return m
    if name == 'resnet18':
        m = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT
            )
        m.fc = nn.Linear(m.fc.in_features, K)
        return m
    if name == 'regnet_x_400mf':
        m = torchvision.models.regnet_x_400mf(
            weights=torchvision.models.RegNet_X_400MF_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, K)
        return m

# class weights from the TRAIN set only
n = train_df['label'].value_counts().sort_index().values
w = (len(train_df) / (K * n)) ** 0.5
criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))

def run_epoch(model, loader, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    loss_sum, preds, labels = 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x); loss = criterion(out, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler: scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                else: loss.backward(); optimizer.step()
            loss_sum += loss.item() * len(y)
            preds.append(out.argmax(1).cpu()); labels.append(y.cpu())
    p, l = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return loss_sum / len(l), accuracy_score(l, p), f1_score(l, p, average='macro')

EPOCHS = {'small_cnn': 15, 'mobilenet_v3_small': 10, 'resnet18': 10,'regnet_x_400mf': 10}
LR = {'small_cnn': 2e-3, 'mobilenet_v3_small': 5e-4, 'resnet18': 5e-4,'regnet_x_400mf':5e-4}

PATIENCE = 3          # stop after 3 epochs without improvement
MIN_DELTA = 0.001     # improvement smaller than this does not count

def fit(name):
    model = build_model(name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR[name], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS[name])
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    hist = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    best_f1, best_state, wait = -1, None, 0

    for ep in range(EPOCHS[name]):
        t0 = time.time()
        tl, ta, _ = run_epoch(model, train_loader, opt, scaler)
        vl, va, vf1 = run_epoch(model, val_loader)
        sched.step()
        hist['train_loss'].append(tl); hist['val_loss'].append(vl); hist['val_f1'].append(vf1)

        if vf1 > best_f1 + MIN_DELTA:
            best_f1, wait = vf1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1

        print(f'{name} {ep + 1}/{EPOCHS[name]} | train loss {tl:.3f} acc {ta:.3f} | val loss {vl:.3f} acc {va:.3f} F1 {vf1:.3f} | wait {wait}/{PATIENCE} | {time.time() - t0:.0f}s')
        if wait >= PATIENCE:
            print(f'Early stopping at epoch {ep + 1}; best val F1 {best_f1:.3f}')
            break

    model.load_state_dict(best_state)
    torch.save(best_state, f'{name}.pt')
    return model, hist

