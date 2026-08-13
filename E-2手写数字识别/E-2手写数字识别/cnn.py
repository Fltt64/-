import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from top5_acc import *
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        #卷积层1{输入通道数=1，输出通道数（卷积核数量）=6，卷积核大小=3，步长=1}
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1)
        # 卷积层2
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1)
        #池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #全连接层
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)  # [batch,784] → [batch,1,28,28]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
print("开始训练...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
H,W,C=28,28,1#原始图像，高28，宽28，单通道
train_loader,test_loader=data_dataloader("mnist_x.txt","mnist_y.txt",batch_size=16)
#学习率
learning_rate=0.001
num_epochs=10

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)
best_acc=0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for images,labels in train_loader:
        images,labels=images.to(device),labels.to(device)
        outputs=model(images)
        loss = criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*images.size(0)
    avg_loss = running_loss / len(train_loader.dataset)
    train_acc = topK_acc(model, train_loader, device)
    test_acc = topK_acc(model, test_loader, device)
    train_str = " ".join([f"Top{k + 1}:{v:.4f}" for k, v in enumerate(train_acc)])
    test_str = " ".join([f"Top{k + 1}:{v:.4f}" for k, v in enumerate(test_acc)])
    print(f"Epoch[{epoch + 1}/{num_epochs}]  "
          f"Loss: {avg_loss:.4f}  "
          f"Train Acc: [{train_str}]  "
          f"Test Acc: [{test_str}]")
    if best_acc < test_acc[0]:
        best_acc = test_acc[0]
        torch.save(model.state_dict(), "model/cnn.pth")
        print(f"新最佳模型已保存，准确率：{best_acc * 100:.2f}%")
    else:
        print("此次训练不如现有模型")
print(f"训练完成，模型准确率：{best_acc * 100:.2f}%")
#调参记录：
#改变学习率
"""
batch_size:32,learning_rate = 0.00001————>acc:89.21%
batch_size:32,learning_rate = 0.0001————>acc:97.56%
batch_size:32,learning_rate = 0.001————>acc:98.75%
batch_size:32,learning_rate = 0.01————>acc:97.80%
batch_size:32,learning_rate =0.1 ————>acc:11.21%
"""
#改变batch
"""
batch_size:128,learning_rate = 0.001————>acc:98.70%
batch_size:64,learning_rate = 0.001————>acc:98.74%
batch_size:32,learning_rate = 0.001————>acc:98.75%
batch_size:16,learning_rate = 0.001————>acc:98.85%
batch_size:8,learning_rate = 0.001————>acc:98.98%
"""