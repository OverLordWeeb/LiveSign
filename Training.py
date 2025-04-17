import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from torch.multiprocessing import set_start_method
import matplotlib.pyplot as plt

try:
    set_start_method('spawn')
except RuntimeError:
    pass

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.backends.cudnn.benchmark = True

    dataset_path = "C:\\Users\\pkucz\\Desktop\\datasetS"

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    letters_path = os.path.join(dataset_path, "Letters")
    numbers_path = os.path.join(dataset_path, "Numbers")

    letters_dataset = datasets.ImageFolder(root=letters_path, transform=transform)
    numbers_dataset = datasets.ImageFolder(root=numbers_path, transform=transform)

    full_dataset = torch.utils.data.ConcatDataset([letters_dataset, numbers_dataset])

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    num_classes = len(letters_dataset.classes) + len(numbers_dataset.classes)
    print(f"Total Classes: {num_classes}")

    class SignLanguageModel(nn.Module):
        def __init__(self, num_classes):
            super(SignLanguageModel, self).__init__()
            self.conv_layers = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.fc_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 8 * 8, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )

        def forward(self, x):
            x = self.conv_layers(x)
            x = self.fc_layers(x)
            return x

    model = SignLanguageModel(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler()

    checkpoint_path = "sign_language_checkpoint.pth"
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed training from epoch {start_epoch}")
    else:
        print("Starting new training session")

    criterion = nn.CrossEntropyLoss()

    train_losses = []
    train_accuracies = []
    val_accuracies = []

    plt.ion()
    fig, ax = plt.subplots()
    plot_initialized = False

    while True:
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{start_epoch}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{start_epoch}] complete. Average Loss: {avg_loss:.4f}")

        model.eval()
        train_correct = 0
        train_total = 0
        with torch.no_grad():
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

        train_accuracy = 100 * train_correct / train_total
        print(f"Epoch [{start_epoch}] Training Accuracy: {train_accuracy:.2f}%")

        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        print(f"Epoch [{start_epoch}] Validation Accuracy: {val_accuracy:.2f}%")

        train_losses.append(avg_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        ax.clear()
        ax.plot(train_losses, label="Train Loss")
        ax.plot(train_accuracies, label="Train Accuracy")
        ax.plot(val_accuracies, label="Validation Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.set_title("Training Progress")
        ax.legend()

        if not plot_initialized:
            plt.show(block=False)
            plot_initialized = True

        fig.canvas.draw()
        fig.canvas.flush_events()

        torch.save({
            "epoch": start_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict()
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {start_epoch}")

        start_epoch += 1

if __name__ == "__main__":
    main()
