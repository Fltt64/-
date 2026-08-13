from ResNet import ResNet,test_loader
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet().to(device)
model.load_state_dict(torch.load("model/ResNet.pth", map_location=device))
criterion = torch.nn.CrossEntropyLoss()
model.eval()
test_running_loss = 0.0
test_correct = 0
test_total = 0
print("开始测试")
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
    epoch_test_loss = test_running_loss / test_total
    epoch_test_acc = test_correct / test_total
print("test_loss: ", epoch_test_loss)
print(f"test_acc:{epoch_test_acc*100}%")