from MLP import MLP
from utils import *
stats = np.load("model/stats.npz")
loader, passenger_ids = predict_dataloader("test.csv", stats["mean"], stats["std"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)
model.load_state_dict(torch.load("model/mlp.pth", map_location=device))
model.eval()

preds_list = []
with torch.no_grad():
    print("开始预测...")
    for features, _ in loader:
        features = features.to(device)
        outputs = model(features)
        pred = (torch.sigmoid(outputs) > 0.5).int()
        preds_list.append(pred.cpu())
preds = torch.cat(preds_list).numpy().flatten()
if passenger_ids is not None:
    submission = pd.DataFrame({"PassengerId": passenger_ids,'Survived':preds})
    submission.to_csv("submission.csv", index=False)
    print("submission.csv 已经保存")
