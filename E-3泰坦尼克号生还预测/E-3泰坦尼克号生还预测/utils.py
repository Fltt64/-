import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader,random_split
import pandas as pd
def data_dataloader(data_path,proportion=0.85,batch_size=64,shuffle=True,num_workers=0):
    df=pd.read_csv(data_path)
    features=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
    target='Survived'
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Sex']=df['Sex'].map({'male':1,'female':0})
    df['Embarked'] =df['Embarked'].map({'S':0,'C':1,'Q':2})
    features_np=df[features].values.astype(np.float32)
    targets_np=df[target].values.astype(np.float32)

    # 标准化：z-score = (x - mean) / std
    mean = features_np.mean(axis=0)
    std = features_np.std(axis=0)
    std[std == 0] = 1e-8  # 防止除零（常数特征）
    features_np = (features_np - mean) / std

    features_tensor=torch.from_numpy(features_np)
    target_tensor=torch.from_numpy(targets_np).unsqueeze(1)

    dataset = TensorDataset(features_tensor,target_tensor)

    ##训练集与测试集的划分(默认4：1）
    total = len(dataset)
    train_len = int(total * proportion)
    test_len = total - train_len
    train_dataset, test_dataset = random_split(dataset, [train_len, test_len])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return train_loader, test_loader, mean, std

def predict_dataloader(data_path, mean, std, batch_size=64, num_workers=0):
    df = pd.read_csv(data_path)

    passenger_ids = df['PassengerId'].values
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    features_np = df[features].values.astype(np.float32)
    features_np = (features_np - mean) / std

    features_tensor = torch.from_numpy(features_np)
    dummy_target = torch.zeros(len(features_tensor), 1)  # test.csv 无标签，占位

    dataset = TensorDataset(features_tensor, dummy_target)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader, passenger_ids

def data_dataloader_rf(data_path):
    df=pd.read_csv(data_path)
    features=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
    target='Survived'
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Sex']=df['Sex'].map({'male':1,'female':0})
    df['Embarked'] =df['Embarked'].map({'S':0,'C':1,'Q':2})
    features_np=df[features].values.astype(np.float32)
    targets_np=df[target].values.astype(np.float32)

    # 标准化：z-score = (x - mean) / std
    mean = features_np.mean(axis=0)
    std = features_np.std(axis=0)
    std[std == 0] = 1e-8  # 防止除零（常数特征）
    features_np = (features_np - mean) / std
    return features_np, targets_np

def predict_dataloader_rf(data_path, mean, std):
    df = pd.read_csv(data_path)

    passenger_ids = df['PassengerId'].values
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    features_np = df[features].values.astype(np.float32)
    features_np = (features_np - mean) / std
    return features_np, passenger_ids