import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
def train_model(train_loader, val_loader, model, criterion, optimizer,model_name,scheduler=None, num_epochs=20, device="cuda"):
    model = model.to(device)
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    best_acc = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        # 计算本轮平均损失和准确率
        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():  # 验证不计算梯度，省显存
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        epoch_val_loss = val_running_loss / val_total
        epoch_val_acc = val_correct / val_total
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        if best_acc < epoch_val_acc and model_name=="VGG":
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), "model/VGG.pth")
            print(f"新最佳模型已保存，准确率：{best_acc * 100:.2f}%")
        elif best_acc >= epoch_val_acc and model_name=="VGG":
            print("此次训练不如现有模型")
        elif best_acc < epoch_val_acc and model_name=="ResNet":
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), "model/ResNet.pth")
            print(f"新最佳模型已保存，准确率：{best_acc * 100:.2f}%")
        elif best_acc >= epoch_val_acc and model_name=="ResNet":
            print("此次训练不如现有模型")
        if scheduler is not None:
            scheduler.step()
    print("训练完成！")
    return history