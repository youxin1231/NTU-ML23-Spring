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
from tqdm import tqdm, trange
from scipy import stats
import copy

from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from torchvision.models import resnet50, resnet152, resnext101_64x4d, wide_resnet101_2, densenet201
from torchensemble import VotingClassifier
from torchensemble.utils import io

myseed = 2023  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

"""# Transforms
Torchvision provides lots of useful utilities for image preprocessing, data *wrapping* as well as data augmentation.

Please refer to PyTorch official website for details about different transforms.
"""

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

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

# Kfolds
k_folds = 5

# The number of batch size.
batch_size = 64

# Test time augmentation ratio
test_tfm_ratio = 0.85

"""# Models"""

model_archs = {
    # 'wide_resnet101': wide_resnet101_2(weights=None).to(device),
    # 'resnext101': resnext101_64x4d(weights=None).to(device),
    'densenet201': densenet201(weights=None).to(device),
    # 'resnet50': resnet50(weights=None).to(device),
    # 'resnet152': resnet152(weights=None).to(device),
}

"""# Dataloader for test"""

# Construct test datasets.
# The argument "loader" tells how torchvision reads the data.
test_set = FoodDataset("./data/test", tfm=test_tfm)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

test_set_w_train_tfm = FoodDataset("./data/test", tfm=train_tfm)
test_loader_w_train_tfm = DataLoader(test_set_w_train_tfm, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# Ensemble
models = []
for model_key, model_arch in model_archs.items():
    for fold in range(k_folds):
        model_path = f"./ckpt/{model_key}_{fold}.ckpt"
        if os.path.isfile(model_path):
            model = copy.deepcopy(model_arch)
            model.load_state_dict(torch.load(model_path))
            models.append(model)

    # ensemble = VotingClassifier(
    #     estimator=model,
    #     n_estimators=5,
    # )
    # ensemble.set_criterion(nn.CrossEntropyLoss())
    # ensemble.set_optimizer(
    #     "AdamW",                                 # type of parameter optimizer
    #     lr=lr,                       # learning rate of parameter optimizer
    #     weight_decay=weight_decay,              # weight decay of parameter optimizer
    # )
    # ensemble.set_scheduler(
    #     "CosineAnnealingLR",                    # type of learning rate scheduler
    #     T_max=n_epochs,                           # additional arguments on the scheduler
    # )
    # io.load(ensemble, f"./ckpt/{_exp_name}_{i}.ckpt")  # reload

"""# Testing and generate prediction CSV"""

prediction_logits = np.zeros((3000, 1000))
prediction_votings = [[] for i in range(len(model_archs) * k_folds)]

with torch.no_grad():
    for i, model in enumerate(tqdm(models)):
        test_tfm_list = []
        train_tfm_list = []

        for data, _ in test_loader:
            test_w_test_tfm_pred = model(data.to(device))
            test_tfm_list += test_w_test_tfm_pred.squeeze().tolist()
        
        for data, _ in test_loader_w_train_tfm:
            test_w_train_tfm_pred = model(data.to(device))
            train_tfm_list += test_w_train_tfm_pred.squeeze().tolist()
        
        test_pred = np.array(test_tfm_list) * test_tfm_ratio + np.array(train_tfm_list) * (1 - test_tfm_ratio)
        test_label = np.argmax(test_pred, axis=1)

        prediction_logits += test_pred
        prediction_votings[i] += test_label.squeeze().tolist()

prediction_logit = np.argmax(prediction_logits, axis=1)

prediction_voting, count = stats.mode(np.array(prediction_votings))
prediction_voting = prediction_voting[0]

prediction = prediction_logit
# prediction = prediction_voting

# create test csv
def pad4(i):
    return "0"*(4-len(str(i)))+str(i)
df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(len(test_set))]
df["Category"] = prediction
df.to_csv("submission.csv",index = False)

"""# Q1. Augmentation Implementation
## Implement augmentation by finishing train_tfm in the code with image size of your choice. 
## Directly copy the following block and paste it on GradeScope after you finish the code
### Your train_tfm must be capable of producing 5+ different results when given an identical image multiple times.
### Your  train_tfm in the report can be different from train_tfm in your training code.

"""

train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize(size=(128, 128)),
    # You may add some transforms here.
    transforms.RandomRotation(degrees=(0, 180)), # Randomly rotate the image between 0 and 180 degrees.
    transforms.RandomHorizontalFlip(), # Randomly flip the image horizontally.
    transforms.RandomPerspective(), # Random perspective transformation.
    transforms.ColorJitter(), # Randomly change the contrast, saturation, and hue of image.
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

"""# Q2. Visual Representations Implementation
## Visualize the learned visual representations of the CNN model on the validation set by implementing t-SNE (t-distributed Stochastic Neighbor Embedding) on the output of both top & mid layers (You need to submit 2 images). 

"""

# Load the trained model
intermediate_layer = {
    'features.norm5': 'top_layer',
    'features.transition2.pool': 'mid_layer',
}
model = densenet201(weights=None).to(device)
model.load_state_dict(torch.load(f"./ckpt/densenet201_0.ckpt"))

model = create_feature_extractor(model, return_nodes=intermediate_layer)

model.eval()

# Load the vaildation set defined by TA
valid_set = FoodDataset("./data/valid", tfm=test_tfm)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# Extract the representations for the specific layer of model
# You should find out the index of layer which is defined as "top" or 'mid' layer of your model.

for feat in intermediate_layer.values():
    features = []
    labels = []
    for batch in tqdm(valid_loader):
        imgs, lbls = batch
        with torch.no_grad():
            logits = model(imgs.to(device))[feat]
            logits = logits.view(logits.size()[0], -1)
        labels.extend(lbls.cpu().numpy())
        logits = np.squeeze(logits.cpu().numpy())
        features.extend(logits)
        
    features = np.array(features)
    colors_per_class = cm.rainbow(np.linspace(0, 1, 11))

    # Apply t-SNE to the features
    features_tsne = TSNE(n_components=2, init='pca', random_state=42).fit_transform(features)

    # Plot the t-SNE visualization
    plt.figure(figsize=(10, 8))
    for label in np.unique(labels):
        plt.scatter(features_tsne[labels == label, 0], features_tsne[labels == label, 1], label=label, s=5)
    plt.legend()
    plt.savefig(f'{feat}.png')