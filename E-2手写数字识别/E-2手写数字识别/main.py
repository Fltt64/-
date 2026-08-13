import torch.nn as nn
import torch.optim as optim
from utils import *
from top5_acc import *
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(28*28,512)
        self.fc2 = nn.Linear(512,128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64,10)
    def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
print("开始训练...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
learning_rate = 0.001#学习率
num_epochs = 10#循环次数

model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

train_loader,test_loader=data_dataloader("mnist_x.txt","mnist_y.txt",batch_size=32)
best_acc=0#模型最佳准确率
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for (images,labels) in train_loader:
        images,labels = images.to(device),labels.to(device)
        outputs = model(images)
        loss = criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    train_acc = topK_acc(model, train_loader, device)
    test_acc = topK_acc(model, test_loader, device)
    avg_loss=running_loss/len(train_loader)
    train_str = " ".join([f"Top{k+1}:{v:.4f}" for k, v in enumerate(train_acc)])
    test_str = " ".join([f"Top{k+1}:{v:.4f}" for k, v in enumerate(test_acc)])
    print(f"Epoch[{epoch+1}/{num_epochs}]  "
          f"Loss: {avg_loss:.4f}  "
          f"Train Acc: [{train_str}]  "
          f"Test Acc: [{test_str}]")
    if best_acc < test_acc[0]:
        best_acc = test_acc[0]
        torch.save(model.state_dict(), "model/mlp.pth")
        print(f"新最佳模型已保存，准确率：{best_acc * 100:.2f}%")
    else:
        print("此次训练不如现有模型")
print(f"训练完成，模型准确率：{best_acc*100:.2f}%")
#调参记录：
#改变batch
"""
batch_size:128,learning_rate = 0.001————>acc:97.79%
batch_size:64,learning_rate = 0.001————>acc:97.95%
batch_size:32,learning_rate = 0.001————>acc:98.09%
batch_size:16,learning_rate = 0.001————>acc:98.06%
batch_size:8,learning_rate = 0.001————>acc:97.89%
"""
#改变学习率
"""
batch_size:32,learning_rate = 0.00001————>acc:91.66%
batch_size:32,learning_rate = 0.0001————>acc:97.48%
batch_size:32,learning_rate = 0.001————>acc:97.99%
batch_size:32,learning_rate = 0.01————>acc:96.63%
batch_size:32,learning_rate = 0.1————>acc:11.52%
"""