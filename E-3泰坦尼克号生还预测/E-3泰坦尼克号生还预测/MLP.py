import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import *
from acc import *
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(7,512)
        self.fc2 = nn.Linear(512,128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64,32)
        self.fc5 = nn.Linear(32,16)
        self.fc6 = nn.Linear(16,1)
    def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)
        return x

if __name__ == '__main__':
    print("开始训练...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    learning_rate = 0.001  # 学习率
    num_epochs = 15  # 循环次数

    model = MLP().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loader, test_loader, mean, std = data_dataloader("train.csv", batch_size=8)
    np.savez("model/stats.npz", mean=mean, std=std)
    print(f"mean/std 已保存至 model/stats.npz")
    best_acc = 0  # 模型最佳准确率
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for (images, labels) in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_acc = accuracy(model, train_loader, device)
        test_acc = accuracy(model, test_loader, device)
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch[{epoch + 1}/{num_epochs}]  "
              f"Loss: {avg_loss:.4f}  "
              f"Train Acc: {train_acc:.4f}  "
              f"Test Acc: {test_acc:.4f}")
        if best_acc < test_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "model/mlp.pth")
            print(f"新最佳模型已保存，准确率：{best_acc * 100:.2f}%")
        else:
            print("此次训练不如现有模型")
    print(f"训练完成，模型准确率：{best_acc * 100:.2f}%")