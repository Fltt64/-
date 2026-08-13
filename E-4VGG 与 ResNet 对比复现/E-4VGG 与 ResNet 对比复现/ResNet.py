import warnings
warnings.filterwarnings("ignore")
import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from utils import train_model
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
#BasicBlock残差块
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            x = self.downsample(x)
        out=out + x
        return self.relu(out)
#创造 BasicBlock残差块
def resnet_block(in_channels, out_channels,num_blocks):
    layers = []
    for _ in range(num_blocks):
        if _ == 0:
            stride=2
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))
            layers.append(BasicBlock(in_channels, out_channels, stride,downsample))
        else:
            stride=1
            layers.append(BasicBlock(out_channels, out_channels, stride, downsample=None))
    return nn.Sequential(*layers)

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1=resnet_block(3,64,3)
        self.block2=resnet_block(64,128,3)
        self.block3=resnet_block(128,256,3)
        self.block4=resnet_block(256,512,3)
        self.fc1 = nn.Linear(2048, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 10)
        # 全连接层正则化,丢弃率0.5
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        #卷积层
        out = self.block4(self.block3(self.block2(self.block1(x))))
        out = out.view(x.size(0), -1)
        # 全连接层
        out = F.relu(self.bn_fc1(self.fc1(out)))
        out = self.dropout(out)
        out = F.relu(self.bn_fc2(self.fc2(out)))
        out = self.dropout(out)
        out= self.fc3(out)
        return out
#数据预处理
batch_size=64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize( mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616))
])
train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=transform)
test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=transform)
train_size=45000
val_size=5000
train_set,val_set=torch.utils.data.random_split(train_dataset,[train_size,val_size])

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0)
val_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0)
if __name__ == '__main__':
    print("=" * 50)
    print("训练开始...")
    print("=" * 50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    num_epochs = 30
    model = ResNet().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], gamma=0.1)
    resnet_history = train_model(train_loader, val_loader,model_name="ResNet", model=model, criterion=criterion, optimizer=optimizer,
                              scheduler=scheduler, num_epochs=num_epochs, device=device)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, num_epochs + 1)
    #左图：Loss曲线
    ax1.plot(epochs, resnet_history["train_loss"], 'b-o', markersize=5, label='Train Loss')
    ax1.plot(epochs, resnet_history["val_loss"], 'r-s', markersize=5, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('ResNet Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # 右图：Accuracy曲线
    ax2.plot(epochs, resnet_history["train_acc"], 'b-o', markersize=5, label='Train Acc')
    ax2.plot(epochs, resnet_history["val_acc"], 'r-s', markersize=5, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('ResNet Training & Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    best_epoch = max(range(num_epochs), key=lambda i: resnet_history["val_acc"][i])
    fig.suptitle(f'ResNet on CIFAR-10  |  Best Val Acc: {resnet_history["val_acc"][best_epoch]:.2%} @ Epoch {best_epoch + 1}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("resnet_history.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n图片已保存至 resnet_history.png")