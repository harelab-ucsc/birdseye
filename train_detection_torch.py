
"Pytorch script with MobileNetV2 (download pretrained weights)"
"Make sure to adjust/confirm text files with pixel coordinates"

import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchvision import models

IMG_WIDTH, IMG_HEIGHT = 512, 384
BATCH_SIZE = 16
EPOCHS = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PointDataset(Dataset):
    def __init__(self, image_folder, label_file, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.labels = {}


        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    img_name, x, y = parts[0], float(parts[1]), float(parts[2])
                    self.labels[img_name] = (x, y)

        self.image_files = [os.path.join(image_folder, fname) for fname in self.labels.keys()]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img_name = os.path.basename(img_path)
        x, y = self.labels[img_name]

        image = Image.open(img_path).convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT))
        if self.transform:
            image = self.transform(image)

        label = torch.tensor([x / IMG_WIDTH, y / IMG_HEIGHT], dtype=torch.float32)
        return image, label


transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])


image_folder = os.path.expanduser("~/farm_0_20240725")  # 👈 Change this if needed
label_file = os.path.join(image_folder, "labels.txt")   # 👈 Should match shared file name

dataset = PointDataset(image_folder, label_file, transform)
train_set, test_set = train_test_split(dataset, test_size=0.3, shuffle=True)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

# MOBILENETV2 MODEL
class PointTrackerMobileNet(nn.Module):
    def __init__(self):
        super(PointTrackerMobileNet, self).__init__()
        self.backbone = models.mobilenet_v2(pretrained=True)
        self.backbone.classifier = nn.Sequential(
            nn.Linear(self.backbone.last_channel, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.backbone(x)

model = PointTrackerMobileNet().to(device)

# Training
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

# ~~~~~~~~~~~~~~~~~~~~~
# VISUALIZE PREDICTIONS
# ~~~~~~~~~~~~~~~~~~~~~
def denormalize_coords(coords):
    return coords * torch.tensor([IMG_WIDTH, IMG_HEIGHT], device=coords.device)

model.eval()
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs)

        for i in range(min(3, imgs.size(0))):
            img = imgs[i].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
            true = denormalize_coords(labels[i]).cpu().numpy()
            pred = denormalize_coords(preds[i]).cpu().numpy()

            plt.imshow(img)
            plt.scatter(*true, c='green', label='True')
            plt.scatter(*pred, c='red', label='Predicted')
            plt.legend()
            plt.title("Green = True, Red = Predicted")
            plt.show()
