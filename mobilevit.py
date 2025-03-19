import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import random
import math
from PIL import Image

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

def get_data_loaders(batch_size=64, image_size=224):
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader

train_loader, val_loader = get_data_loaders(batch_size=64, image_size=224)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(ConvNormAct, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor=4):
        super(InvertedResidual, self).__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = int(in_channels * expansion_factor)

        layers = []
        if expansion_factor != 1:
            layers.append(ConvNormAct(in_channels, hidden_dim, kernel_size=1))

        layers.extend([
            ConvNormAct(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.layers(x)
        else:
            return self.layers(x)

class MobileViTBlock(nn.Module):
    def __init__(self, in_channels, d_model, patch_size=2, transformer_depth=2, num_heads=4):
        super(MobileViTBlock, self).__init__()
        self.patch_h, self.patch_w = patch_size, patch_size

        self.local_rep = nn.Sequential(
            ConvNormAct(in_channels, in_channels, kernel_size=3, padding=1),
            ConvNormAct(in_channels, d_model, kernel_size=1)
        )

        transformer_layers = []
        for _ in range(transformer_depth):
            transformer_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=d_model * patch_size * patch_size,
                    nhead=num_heads,
                    dim_feedforward=d_model * patch_size * patch_size * 2,
                    dropout=0.0,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True
                )
            )
        self.transformer = nn.Sequential(*transformer_layers)

        self.proj = ConvNormAct(d_model, in_channels, kernel_size=1)
        self.fusion = ConvNormAct(2*in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        batch_size, in_channels, height, width = x.shape
        residual = x

        local_rep = self.local_rep(x)

        patch_h, patch_w = self.patch_h, self.patch_w
        d_model = local_rep.shape[1]

        new_h = int(math.ceil(height / patch_h) * patch_h)
        new_w = int(math.ceil(width / patch_w) * patch_w)

        if new_h != height or new_w != width:
            padding_h = new_h - height
            padding_w = new_w - width
            local_rep = F.pad(local_rep, (0, padding_w, 0, padding_h))

        num_patch_h = new_h // patch_h
        num_patch_w = new_w // patch_w
        num_patches = num_patch_h * num_patch_w

        local_rep = local_rep.reshape(batch_size, d_model, num_patch_h, patch_h, num_patch_w, patch_w)
        local_rep = local_rep.permute(0, 2, 4, 3, 5, 1).reshape(batch_size, num_patches, -1)

        global_rep = self.transformer(local_rep)

        global_rep = global_rep.reshape(batch_size, num_patch_h, num_patch_w, patch_h, patch_w, d_model)
        global_rep = global_rep.permute(0, 5, 1, 3, 2, 4).reshape(batch_size, d_model, new_h, new_w)

        if new_h != height or new_w != width:
            global_rep = global_rep[:, :, :height, :width]

        global_rep = self.proj(global_rep)
        output = self.fusion(torch.cat([residual, global_rep], dim=1))

        return output

class MobileViT(nn.Module):
    def __init__(self, image_size=224, dims=[96, 120, 144], expansion=4, kernel_size=3, patch_size=2, num_classes=10):
        super(MobileViT, self).__init__()

        if dims == [96, 120, 144]:  # MobileViT-S
            channels = [16, 32, 64, 96, 128, 160, 640]
            transformer_depth = [2, 4, 3]
        elif dims == [64, 80, 96]:  # MobileViT-XS
            channels = [16, 32, 48, 64, 80, 96, 384]
            transformer_depth = [2, 4, 3]
        elif dims == [32, 64, 80]:  # MobileViT-XXS
            channels = [16, 16, 24, 48, 64, 80, 320]
            transformer_depth = [2, 4, 3]
        else:
            raise ValueError("Unsupported MobileViT configuration")

        self.conv1 = ConvNormAct(3, channels[0], kernel_size=3, stride=2, padding=1)

        self.mv2 = nn.ModuleList([
            InvertedResidual(channels[0], channels[1], stride=1, expansion_factor=expansion),
            InvertedResidual(channels[1], channels[2], stride=2, expansion_factor=expansion),
            InvertedResidual(channels[2], channels[2], stride=1, expansion_factor=expansion),
            InvertedResidual(channels[2], channels[3], stride=2, expansion_factor=expansion),
        ])

        self.mobilevit_stage1 = nn.Sequential(
            MobileViTBlock(channels[3], dims[0], patch_size=patch_size, transformer_depth=transformer_depth[0]),
            InvertedResidual(channels[3], channels[4], stride=2, expansion_factor=expansion)
        )

        self.mobilevit_stage2 = nn.Sequential(
            MobileViTBlock(channels[4], dims[1], patch_size=patch_size, transformer_depth=transformer_depth[1]),
            InvertedResidual(channels[4], channels[5], stride=2, expansion_factor=expansion)
        )

        self.mobilevit_stage3 = MobileViTBlock(channels[5], dims[2], patch_size=patch_size, transformer_depth=transformer_depth[2])

        self.conv_last = ConvNormAct(channels[5], channels[6], kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels[6], num_classes)

    def forward(self, x):
        x = self.conv1(x)

        for layer in self.mv2:
            x = layer(x)

        x = self.mobilevit_stage1(x)
        x = self.mobilevit_stage2(x)
        x = self.mobilevit_stage3(x)

        x = self.conv_last(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

model = MobileViT(dims=[32, 64, 80], num_classes=10).to(device)
print(model)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Trainable parameters: {count_parameters(model):,}")

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')

        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({'loss': loss.item(), 'acc': 100. * train_correct / train_total})

        scheduler.step()

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = 100. * train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100. * val_correct / val_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'mobilevit_best.pth')

        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    torch.save(model.state_dict(), 'mobilevit_last.pth')

    return history

epochs = 10
history = train_model(model, train_loader, val_loader, epochs=epochs, lr=0.001)

def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            c = predicted.eq(targets).cpu().numpy()
            for i in range(targets.size(0)):
                label = targets[i].item()
                class_correct[label] += c[i]
                class_total[label] += 1

    acc = 100. * correct / total
    print(f'Overall accuracy: {acc:.2f}%')

    for i in range(10):
        print(f'Class {classes[i]} accuracy: {100 * class_correct[i] / class_total[i]:.2f}%')

    return acc

def plot_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('loss_accuracy_history.png')
    plt.show()

model.load_state_dict(torch.load('mobilevit_best.pth'))

val_acc = evaluate_model(model, val_loader)

plot_history(history)

def visualize_predictions(model, data_loader, num_images=5):
    model.eval()

    data_iter = iter(data_loader)
    images, labels = next(data_iter)

    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    plt.figure(figsize=(12, 8))

    for i in range(num_images):
        plt.subplot(2, 5, i + 1)
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        plt.imshow(img)
        plt.title(f'Pred: {classes[preds[i]]}\nTrue: {classes[labels[i]]}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def print_model_size_comparison():
    sizes = {
        "MobileNetV2": count_parameters(models.mobilenet_v2(num_classes=10)),
        "MobileViT-XXS": count_parameters(MobileViT(dims=[32, 64, 80], num_classes=10)),
        "MobileViT-XS": count_parameters(MobileViT(dims=[64, 80, 96], num_classes=10)),
        "ResNet-18": count_parameters(models.resnet18(num_classes=10)),
        "MobileViT-S": count_parameters(MobileViT(dims=[96, 120, 144], num_classes=10))
    }

    for name, size in sorted(sizes.items(), key=lambda x: x[1]):
        print(f"{name}: {size:,} parameters")

visualize_predictions(model, val_loader, num_images=10)
print_model_size_comparison()