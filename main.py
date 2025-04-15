import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    handlers=[
        logging.FileHandler("training.log"),  # 将日志写入文件
        logging.StreamHandler()  # 同时输出到控制台
    ]
)

# CNN with BatchNorm, Dropout, Xavier init, Adam, EarlyStopping
class StockCNN(nn.Module):
    def __init__(self):
        super(StockCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(5,3), padding=(2,1))
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(5,3), padding=(2,1))
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=(5,3), padding=(2,1))
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=(5,3), padding=(2,1))
        self.bn4 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(kernel_size=(2,1))
        self.lrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.5)

        self.fc = nn.Linear(512 * 6 * 180, 2)  

        self._initialize_weights()

    def forward(self, x):
        x = self.pool(self.lrelu(self.bn1(self.conv1(x))))
        x = self.pool(self.lrelu(self.bn2(self.conv2(x))))
        x = self.pool(self.lrelu(self.bn3(self.conv3(x))))
        x = self.pool(self.lrelu(self.bn4(self.conv4(x))))

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)




if __name__=="__main__":
    # Dataset transforms
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.229, 0.224, 0.225],  [0.485, 0.456, 0.406])
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.229, 0.224, 0.225],  [0.485, 0.456, 0.406])
    ])

    # Load dataset
    train_dataset = datasets.ImageFolder(root='./data/image/train', transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=1)
    test_dataset = datasets.ImageFolder(root='./data/image/test', transform=test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=1)

    for batch_num, (feats, labels) in enumerate(train_dataloader):
        print(feats.shape)
        print(labels)
        break
    
    # Training config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StockCNN().to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # Early stopping params
    early_stop_patience = 2
    best_loss = float('inf')
    patience_counter = 0
    num_epochs = 100
    train_loss_list, acc_list = [], []

    logging.info("Training...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in train_dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            accuracy_rate = torch.sum(torch.argmax(outputs, dim=1) == labels).item() / len(labels)
            logging.info(f"Epoch {epoch+1}, Batch {batch_num+1}: Loss={loss.item():.4f}, Accuracy={accuracy_rate:.4f}")
            
            
            total_loss += loss.item() * imgs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        acc = correct / total
        train_loss_list.append(avg_loss)
        acc_list.append(acc)

        logging.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={acc:.4f}")
        

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logging.info("Early stopping triggered.")
                break

    # Final evaluation
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for imgs, labels in test_dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    logging.info(f"Test Accuracy: {correct / total:.4f}")

    # Plot loss and accuracy
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(train_loss_list, label="Train Loss")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(acc_list, label="Train Accuracy")
    plt.legend()
    plt.show()
    logging.info("Finished training!")