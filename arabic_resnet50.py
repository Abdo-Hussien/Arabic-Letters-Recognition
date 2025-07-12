"""arabic_resnet50.py
Arabic Letter Recognition with ResNet‑50 (PyTorch).

Expected directory structure *after* extraction
└── Computer_Vision
    └── Computer Vision
        ├── Train Images 13440x32x32
        │   └── train
        └── Test Images 3360x32x32
            └── test

If running on Google Colab, the script mounts Google Drive, extracts the
dataset ZIP, and then trains & evaluates a fine‑tuned ResNet‑50.
"""


def main():
    # ENVIRONMENT & DATA I/O
    import os
    import re
    import zipfile
    from pathlib import Path

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, models
    from PIL import Image

    # ----- Colab‑specific: mount Google Drive -----
    try:
        from google.colab import drive  # type: ignore

        IN_COLAB = True
    except ImportError:
        IN_COLAB = False

    if IN_COLAB:
        drive.mount('/content/drive')

    # Path to the ZIP file in Drive (edit as needed)
    ZIP_PATH = "Computer Vision.zip"

    # Where to extract
    EXTRACT_ROOT = Path("/content/Computer_Vision")
    (EXTRACT_ROOT).mkdir(parents=True, exist_ok=True)

    # Extract only if the target dir does not already exist
    if not (EXTRACT_ROOT / "Computer Vision").exists():
        with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
            zf.extractall(EXTRACT_ROOT)
            print("✅ Dataset extracted to", EXTRACT_ROOT.resolve())
    else:
        print("📂 Dataset already extracted ->", EXTRACT_ROOT)

    # Train/Test folders
    TRAIN_DIR = EXTRACT_ROOT / "Computer Vision/Train Images 13440x32x32/train"
    TEST_DIR = EXTRACT_ROOT / "Computer Vision/Test Images 3360x32x32/test"

    class ArabicLetterDataset(Dataset):
        """Dataset that reads PNG images named *label_XX.png* and returns (img, label)."""

        def __init__(self, root_dir: Path, transform=None):
            self.root_dir = Path(root_dir)
            self.transform = transform
            self.files = sorted([f for f in os.listdir(root_dir) if f.endswith('.png')])

        def __len__(self):
            return len(self.files)

        def __getitem__(self, index):
            img_name = self.files[index]
            match = re.search(r'label_(\d+)', img_name)
            label = int(match.group(1)) - 1 if match else -1  # 0‑based
            img_path = self.root_dir / img_name

            image = Image.open(img_path).convert("L")  # grayscale

            if self.transform:
                image = self.transform(image)

            return image, label

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # HWC->CHW, scale to [0,1]
        transforms.Normalize((0.5,), (0.5,))  # center to [-1,1]
    ])

    # DataLoaders
    train_dataset = ArabicLetterDataset(TRAIN_DIR, transform)
    test_dataset = ArabicLetterDataset(TEST_DIR, transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    print(f"🖼️  Loaded {len(train_dataset)} training and {len(test_dataset)} test images.")

    # MODEL: RESNET‑50 FT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("⚡ Using device:", device)

    # Load pretrained ResNet‑50
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Modify first conv layer to accept 1‑channel input
    # Keep kernel/stride/padding same; just change in_channels
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Replace the final FC layer for 28 classes
    model.fc = nn.Linear(model.fc.in_features, 28)

    # (Optional) Freeze feature extractor; comment these two lines for full fine‑tuning
    # for param in model.parameters():
    #     param.requires_grad = False
    # for param in model.fc.parameters():
    #     param.requires_grad = True

    model.to(device)

    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    class EarlyStopping:
        """Stop training if no loss improvement after *patience* epochs."""

        def __init__(self, patience=3, min_delta=0.):
            self.patience = patience
            self.min_delta = min_delta
            self.best_loss = float('inf')
            self.counter = 0
            self.stop = False

        def __call__(self, loss):
            if loss < self.best_loss - self.min_delta:
                self.best_loss = loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.stop = True

    early_stopping = EarlyStopping(patience=2, min_delta=0.005)
    max_grad_norm = 1.0
    NUM_EPOCHS = 10
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        scheduler.step()
        print(f"Epoch {epoch:02d}/{NUM_EPOCHS} — loss: {epoch_loss:.4f}")
        early_stopping(epoch_loss)
        if early_stopping.stop:
            print("⏹️  Early stopping triggered!")
            break

    # EVALUATE & METRICS
    def evaluate(loader, split="data"):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        acc = 100 * correct / total
        print(f"✅ Accuracy on {split}: {acc:.2f}%")
        return acc

    train_acc = evaluate(train_loader, "train")
    test_acc = evaluate(test_loader, "test")

    # PREDICTIONS
    import pandas as pd

    model.eval()
    predictions = []
    with torch.no_grad():
        for img_name in sorted(os.listdir(TEST_DIR)):
            if not img_name.endswith('.png'):
                continue
            img_path = TEST_DIR / img_name
            img = Image.open(img_path).convert('L')
            img = transform(img).unsqueeze(0).to(device)
            pred = int(model(img).argmax(1).item()) + 1  # back to 1‑based
            predictions.append([img_name, pred])

    df = pd.DataFrame(predictions, columns=["Image", "Predicted_Label"])
    csv_path = "test_predictions.csv"
    df.to_csv(csv_path, index=False)
    print("📄 Submission saved to", csv_path)

    # confusion matrix & class accuracies if true labels are encoded in filenames.
    try:
        from sklearn.metrics import confusion_matrix
        import numpy as np
        import matplotlib.pyplot as plt

        true_labels = []
        pred_labels = []
        for img_name, pred in predictions:
            match = re.search(r'label_(\d+)', img_name)
            if match:
                true_labels.append(int(match.group(1)))
                pred_labels.append(pred)
        if true_labels:
            cm = confusion_matrix(true_labels, pred_labels, labels=range(1, 29))
            print("Confusion matrix shape:", cm.shape)
            # Simple visualization
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted');
            plt.ylabel('True')
            plt.colorbar()
            plt.show()
    except Exception as e:
        print("(Could not plot confusion matrix:", e, ")")


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()  # Optional but good to include
    main()
