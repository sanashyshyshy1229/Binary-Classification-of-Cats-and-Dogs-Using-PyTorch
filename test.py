import torch
from torchvision import transforms
from PIL import Image
import os
import random

from model import SimpleCNN

def test():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("catdog.pth", map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    test_dir = "data/test"
    imgs = os.listdir(test_dir)

    # 随机抽取 50 张
    if len(imgs) >= 50:
        imgs = random.sample(imgs, 50)
    else:
        print("测试集不足 50 张，仅测试全部图片。")

    print(f"\n随机测试 {len(imgs)} 张图片：\n")

    for img_name in imgs:
        img_path = os.path.join(test_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img)
            pred = torch.argmax(output, 1).item()

        print(f"{img_name} --> {'Cat' if pred == 0 else 'Dog'}")
