import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR

from dataset import CatDogDataset
from model import SimpleCNN


def train_model():

    # 图像增强（提高泛化）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor()
    ])

    dataset = CatDogDataset(root_dir="data/train", transform=transform)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    print("开始训练...\n")

    for epoch in range(20):  # 训练 20 轮
        model.train()
        total_loss = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        print(f"Epoch {epoch+1}/20 Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "catdog.pth")
    print("训练完成，模型已保存为 catdog.pth")
