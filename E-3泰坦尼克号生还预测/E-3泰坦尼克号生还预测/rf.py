from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from utils import *
print("开始训练...")
stats = np.load("model/stats.npz")
#特征（X） 标签（Y）
X,Y=data_dataloader_rf("train.csv")
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2)

# 网格搜索调参
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, Y_train)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证得分: {grid_search.best_score_ * 100:.2f}%")

# 用最佳模型预测
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
acc = accuracy_score(Y_test, y_pred)
print(f"准确率:{acc*100:.2f}%")

#使用模型进行预测
loader, passenger_ids = predict_dataloader_rf("test.csv", stats["mean"], stats["std"])
preds = best_rf.predict(loader).astype(int)
if passenger_ids is not None:
    submission = pd.DataFrame({"PassengerId": passenger_ids,'Survived':preds})
    submission.to_csv("submission_RF.csv", index=False)
    print("submission_RF.csv 已经保存")