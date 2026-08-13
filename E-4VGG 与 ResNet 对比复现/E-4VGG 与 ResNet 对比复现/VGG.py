import torch.nn as nn
import torch.nn.functional as F
from utils import *
from utils import train_model
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积块1：2层3×3卷积 + BN，通道数64
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_1   = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2   = nn.BatchNorm2d(64)
        # 卷积块2：2层3×3卷积 + BN，通道数128
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_1   = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_2   = nn.BatchNorm2d(128)
        # 卷积块3：3层3×3卷积 + BN，通道数256
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_1   = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_2   = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_3   = nn.BatchNorm2d(256)
        # 卷积块4：3层3×3卷积 + BN，通道数512
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_1   = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_2   = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_3   = nn.BatchNorm2d(512)
        # 卷积块5：3层3×3卷积 + BN，通道数512
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5_1   = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5_2   = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5_3   = nn.BatchNorm2d(512)
        # 最大池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层正则化,丢弃率0.5
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(512 * 1 * 1, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        # 第1卷积块 + 池化
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool(x)  # 32×32 → 16×16
        # 第2卷积块 + 池化
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool(x)  # 16×16 → 8×8
        # 第3卷积块 + 池化
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = F.relu(self.bn3_3(self.conv3_3(x)))
        x = self.pool(x)  # 8×8 → 4×4
        # 第4卷积块 + 池化
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = F.relu(self.bn4_3(self.conv4_3(x)))
        x = self.pool(x)  # 4×4 → 2×2
        # 第5卷积块 + 池化
        x = F.relu(self.bn5_1(self.conv5_1(x)))
        x = F.relu(self.bn5_2(self.conv5_2(x)))
        x = F.relu(self.bn5_3(self.conv5_3(x)))
        x = self.pool(x)  # 2×2 → 1×1
        x = x.view(x.size(0), -1)
        # 全连接层 (BN → ReLU → Dropout, 最后层不加BN/ReLU)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
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
    model = VGG().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], gamma=0.1)
    vgg_history = train_model(train_loader, val_loader, model_name="VGG",model=model, criterion=criterion, optimizer=optimizer,
                              scheduler=scheduler, num_epochs=num_epochs, device=device)
    epochs = range(1, num_epochs + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    #左图：Loss曲线
    ax1.plot(epochs, vgg_history["train_loss"], 'b-o', markersize=5, label='Train Loss')
    ax1.plot(epochs, vgg_history["val_loss"], 'r-s', markersize=5, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('VGG Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    #右图：Accuracy曲线
    ax2.plot(epochs, vgg_history["train_acc"], 'b-o', markersize=5, label='Train Acc')
    ax2.plot(epochs, vgg_history["val_acc"], 'r-s', markersize=5, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('VGG Training & Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    best_epoch = max(range(num_epochs), key=lambda i: vgg_history["val_acc"][i])
    fig.suptitle(f'VGG on CIFAR-10  |  Best Val Acc: {vgg_history["val_acc"][best_epoch]:.2%} @ Epoch {best_epoch + 1}',fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("vgg_history.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n图片已保存至 vgg_history.png")
