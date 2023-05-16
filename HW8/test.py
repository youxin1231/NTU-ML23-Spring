# Import packages
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler
import pandas as pd
from tqdm import tqdm
import os
import cv2
from train import *

"""# Loading data"""

test = np.load("data/testingset.npy", allow_pickle=True)

print(f"Testing dataset size: {test.shape}")

"""# Inference
Model is loaded and generates its anomaly score predictions.

## Initialize
- dataloader
- model
- prediction file
"""

"""## Random seed
Set the random seed to a certain value for reproducibility.
"""

eval_batch_size = 200
model_types = [model_type]
# model_types = ['multi']

# build testing dataloader
data = torch.tensor(test, dtype=torch.float32)
test_dataset = CustomTensorDataset(data)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(
    test_dataset, sampler=test_sampler, batch_size=eval_batch_size, num_workers=1
)
eval_loss = nn.MSELoss(reduction="none")

# load trained model
model_save_dir = "ckpt"

anomality_list = []
origin_list = []
output_list = []

for model_type in  model_types:
    print(f'{model_type}')
    
    checkpoint_path = os.path.join(model_save_dir, f"best_model_{model_type}.pt")
    model = torch.load(checkpoint_path)
    model.eval()

    # prediction file
    out_file = "prediction.csv"

    anomality = list()

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_dataloader)):
            img = data.float().cuda()
            if model_type in ["multi"]:
                output, logits = model(img)
                print(output)
                loss = logits[:batch_size, 1]

            else:
                if model_type in ["fcn"]:
                    img = img.view(img.shape[0], -1)

                output = model(img)

                if model_type in ["vae"]:
                    output = output[0]

                if model_type in ["fcn"]:
                    loss = eval_loss(output, img).sum(-1)

                else:
                    loss = eval_loss(output, img).sum([1, 2, 3])
                
            img = 255 * (img + 1) / 2
            origin_list.extend(img.reshape(-1, 3, 64, 64).permute(0, 2, 3, 1).cpu())

            output = 255 * (output + 1) / 2
            output_list.extend(output.reshape(-1, 3, 64, 64).permute(0, 2, 3, 1).cpu())

            anomality.append(loss)

    anomality = torch.cat(anomality, axis=0)
    anomality = torch.sqrt(anomality).reshape(len(test), 1).cpu().numpy()
    anomality_list.append(anomality)

for idx, tmp in enumerate(anomality_list):
    if idx == 0:
        anomality = tmp
    else:
        anomality = (anomality + tmp) / 2
df = pd.DataFrame(anomality, columns=["score"])
df.to_csv(out_file, index_label="ID")

# Print model architecture
for name, layer in model.named_children():
    print(name, layer)
    
# Plot image
img_size = 64

num_imgs = 25

origin_big_img = np.zeros((5*img_size, 5*img_size, 3), dtype=np.uint8)
output_big_img = np.zeros((5*img_size, 5*img_size, 3), dtype=np.uint8)

for i in range(num_imgs):
    origin_img = np.array(origin_list[i])
    origin_img = cv2.cvtColor(origin_img, cv2.COLOR_RGB2BGR)

    output_img = np.array(output_list[i])
    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

    row = i // 5
    col = i % 5
    origin_big_img[row*img_size:(row+1)*img_size, col*img_size:(col+1)*img_size, :] = origin_img
    output_big_img[row*img_size:(row+1)*img_size, col*img_size:(col+1)*img_size, :] = output_img

cv2.imwrite('origin.png', origin_big_img)
cv2.imwrite('output.png', output_big_img)