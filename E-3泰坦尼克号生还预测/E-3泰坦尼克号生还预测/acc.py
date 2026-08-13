import torch
def accuracy(model, dataloader, device):
    """计算二分类准确率"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, target in dataloader:
            features, target = features.to(device), target.to(device)
            outputs = model(features)
            pred = (torch.sigmoid(outputs) > 0.5).float()
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total