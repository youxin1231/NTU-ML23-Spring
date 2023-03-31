"""# Import Packages"""

# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
# This is for the progress bar.
from tqdm import tqdm
import copy

from torchvision.models import resnet50, resnet152, resnext101_64x4d, wide_resnet101_2, densenet201, vgg13_bn
from torchensemble import VotingClassifier
from sklearn.model_selection import KFold

myseed = 2023  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

"""# Transforms
Torchvision provides lots of useful utilities for image preprocessing, data *wrapping* as well as data augmentation.

Please refer to PyTorch official website for details about different transforms.
"""

# Normally, We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# However, it is also possible to use augmentation in the testing phase.
# You may use train_tfm to produce a variety of images and then test using ensemble methods
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize(size=(128, 128)),
    # You may add some transforms here.
    transforms.RandomRotation(degrees=(0, 180)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomPerspective(),
    transforms.ColorJitter(),
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

"""# Datasets
The data is labelled by the name, so we load images and label while calling '__getitem__'
"""

class FoodDataset(Dataset):

    def __init__(self,path,tfm=test_tfm,files = None):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files
            
        self.transform = tfm
  
    def __len__(self):
        return len(self.files)
  
    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1 # test has no label
            
        return im, label

"""# Configurations"""

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

# The number of batch size.
batch_size = 64

# The number of training epochs.
n_epochs = 1000

# The number of learning rate
lr = 1e-3
weight_decay = 1e-5

# Kfolds
k_folds = 5

# If no improvement in 'patience' epochs, early stop.
patience = 50

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

# Define the K-fold Cross Validator
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=myseed)


# Ensemble
# ensemble = VotingClassifier(
#     estimator=model,
#     n_estimators=5,
# )
# ensemble.set_criterion(nn.CrossEntropyLoss())
# ensemble.set_optimizer(
#     "AdamW",
#     lr=lr,
#     weight_decay=weight_decay,
# )
# ensemble.set_scheduler(
#     "CosineAnnealingLR",
#     T_max=n_epochs,
# )

"""# Models"""

model_archs = {
    # 'wide_resnet101': wide_resnet101_2(weights=None).to(device),
    # 'resnext101': resnext101_64x4d(weights=None).to(device),
    'densenet201': densenet201(weights=None).to(device),
    # 'resnet50': resnet50(weights=None).to(device),
    # 'resnet152': resnet152(weights=None).to(device),
    # "vgg13": vgg13_bn(weights=None).to(device),
}

"""# Dataloader"""

# Construct train and valid datasets.
# The argument "loader" tells how torchvision reads the data.
train_set = FoodDataset("./data/train", tfm=train_tfm)
valid_set = FoodDataset("./data/valid", tfm=test_tfm)
whole_dataset = ConcatDataset([train_set, valid_set])

# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
# valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

# ckpt folder
if not os.path.exists("ckpt"):
    os.makedirs("ckpt")

"""# Start Training"""

for model_key, model_arch in model_archs.items():
    for fold, (train_ids, valid_ids) in enumerate(kfold.split(whole_dataset)):
        print(f'{model_key}\nFOLD {fold}')
        print('--------------------------------')

        model = copy.deepcopy(model_arch)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Initialize trackers, these are not parameters and should not be changed
        stale = 0
        best_acc = 0

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_ids)
        
        # Define data loaders for training and testing data in this fold
        train_loader = torch.utils.data.DataLoader(
                        whole_dataset, 
                        batch_size=batch_size, sampler=train_subsampler)
        valid_loader = torch.utils.data.DataLoader(
                        whole_dataset,
                        batch_size=batch_size, sampler=valid_subsampler)

        # ensemble.fit(train_loader=train_loader,
        #             epochs=n_epochs,
        #             test_loader=valid_loader,
        #             save_model=True,
        #             save_dir=f"./ckpt/{_exp_name}_{fold}.ckpt")

        for epoch in range(n_epochs):

            # ---------- Training ----------
            # Make sure the model is in train mode before training.
            model.train()

            # These are used to record information in training.
            train_loss = []
            train_accs = []

            for batch in tqdm(train_loader):

                # A batch consists of image data and corresponding labels.
                imgs, labels = batch
                #imgs = imgs.half()
                #print(imgs.shape,labels.shape)

                # Forward the data. (Make sure data and model are on the same device.)
                logits = model(imgs.to(device))

                # Calculate the cross-entropy loss.
                # We don't need to apply softmax before computing cross-entropy as it is done automatically.
                loss = criterion(logits, labels.to(device))

                # Gradients stored in the parameters in the previous step should be cleared out first.
                optimizer.zero_grad()

                # Compute the gradients for parameters.
                loss.backward()

                # Clip the gradient norms for stable training.
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

                # Update the parameters with computed gradients.
                optimizer.step()

                # Compute the accuracy for current batch.
                acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

                # Record the loss and accuracy.
                train_loss.append(loss.item())
                train_accs.append(acc)
                
            train_loss = sum(train_loss) / len(train_loss)
            train_acc = sum(train_accs) / len(train_accs)

            # Print the information.
            print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

            # ---------- Validation ----------
            # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
            model.eval()

            # These are used to record information in validation.
            valid_loss = []
            valid_accs = []

            # Iterate the validation set by batches.
            for batch in tqdm(valid_loader):

                # A batch consists of image data and corresponding labels.
                imgs, labels = batch
                #imgs = imgs.half()

                # We don't need gradient in validation.
                # Using torch.no_grad() accelerates the forward process.
                with torch.no_grad():
                    logits = model(imgs.to(device))

                # We can still compute the loss (but not the gradient).
                loss = criterion(logits, labels.to(device))

                # Compute the accuracy for current batch.
                acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

                # Record the loss and accuracy.
                valid_loss.append(loss.item())
                valid_accs.append(acc)
                #break

            # The average loss and accuracy for entire validation set is the average of the recorded values.
            valid_loss = sum(valid_loss) / len(valid_loss)
            valid_acc = sum(valid_accs) / len(valid_accs)

            # Print the information.
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

            # update logs
            if valid_acc > best_acc:
                with open(f"./ckpt/{model_key}_{fold}_log.txt","a") as f:
                    f.write(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best\n")
            else:
                with open(f"./ckpt/{model_key}_{fold}_log.txt","a") as f:
                    f.write(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}\n")
            
            # save models
            if valid_acc > best_acc:
                print(f"Best model found at epoch {epoch}, saving model")
                torch.save(model.state_dict(), f"./ckpt/{model_key}_{fold}.ckpt") # only save best to prevent output memory exceed error
                best_acc = valid_acc
                stale = 0
            else:
                stale += 1
                if stale > patience:
                    print(f"No improvement {patience} consecutive epochs, early stopping")
                    break
