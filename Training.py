import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


class SignLanguageModel(nn.Module):
    def __init__(self, num_classes):
        super(SignLanguageModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class CustomImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples  
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_path = "captured_dataset"
    supported_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    
    # transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    
    # scans for classes and collect samples
    class_names = []
    samples = []
    for folder_type in ['Letters', 'Numbers']:
        root = os.path.join(dataset_path, folder_type)
        if not os.path.isdir(root):
            continue
        for cls in sorted(os.listdir(root)):
            cls_folder = os.path.join(root, cls)
            if not os.path.isdir(cls_folder):
                continue

            files = [f for f in os.listdir(cls_folder)
                     if os.path.splitext(f.lower())[1] in supported_exts]
            if not files:
                continue  # skip if empty class 
            class_index = len(class_names)
            class_names.append(cls)
            for fname in files:
                samples.append((os.path.join(cls_folder, fname), class_index))

    if not samples:
        raise ValueError("No valid images found in captured_dataset. Please ensure you have images under Letters/ and Numbers/ subfolders.")

    num_classes = len(class_names)
    print(f"Classes detected: {class_names}")
    # save labels
    with open("class_labels.json", "w") as f:
        json.dump(class_names, f)


    full_dataset = CustomImageDataset(samples, transform=transform)
    train_size = int(0.7 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)


    model = SignLanguageModel(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # check pointing
    checkpoint_path = "checkpoint.pth"
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print("Starting training from scratch")

    #training loop
    for epoch in range(start_epoch, start_epoch + 4):
        model.train()
        total_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}: Loss={loss.item():.4f}")
        avg_loss = total_loss / len(train_loader)
        # validation
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        print(f"Epoch {epoch+1} complete | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        #save check point
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)

    # last evaluation
    print("\nFinal evaluation on validation set:")
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    torch.save(model.state_dict(), "sign_language_model.pth")
    print("\nModel saved to sign_language_model.pth")

if __name__ == "__main__":
    main()
