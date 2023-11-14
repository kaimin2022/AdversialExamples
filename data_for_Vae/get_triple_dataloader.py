# (pgd, x, y)
import json
from torch.utils.data import TensorDataset, DataLoader
import os
import torch.optim as optim
import time
import torch.utils.data as Data
import torchvision
import torch.nn as nn
import torch
from torchvision import transforms

device = 'cuda'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
train_data = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
)
test_data = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
)
train_data_loader = Data.DataLoader(train_data, batch_size=1, num_workers=0, shuffle=False)
test_data_loader = Data.DataLoader(test_data, batch_size=1, num_workers=0, shuffle=False)
train_x = []
test_x = []
# ---------------get train_set---------------
for x, _ in train_data_loader:
    x = x.to(device)
    x = x.squeeze(dim=0)
    x = x.cpu()
    x = x.detach().numpy().tolist()
    train_x.append(x)
# with open('./data_for_Vae/PGDtrain_data.json', 'r', encoding='utf-8') as yrf:
#     train_pgd = json.load(yrf)
# with open('./data_for_Vae/PGDtrain_label.json', 'r', encoding='utf-8') as yrf:
#     train_y = json.load(yrf)
with open('../data_for_Vae/PGDtrain_data.json', 'r', encoding='utf-8') as yrf:
    train_pgd = json.load(yrf)
with open('../data_for_Vae/PGDtrain_label.json', 'r', encoding='utf-8') as yrf:
    train_y = json.load(yrf)
train_pgd = torch.tensor(train_pgd)
train_x = torch.tensor(train_x)
train_y = torch.tensor(train_y, dtype=torch.int64)
# print(train_pgd.shape, train_x.shape, train_y.shape)
train_set = TensorDataset(train_pgd, train_x, train_y)
# ----------------------- get test set ---------------
for x,_ in test_data_loader:
    x = x.to(device)
    x = x.squeeze(dim=0)
    x = x.cpu()
    x = x.detach().numpy().tolist()
    test_x.append(x)
# with open('./data_for_Vae/PGDtest_data.json', 'r', encoding='utf-8') as yrf:
#     test_pgd = json.load(yrf)
# with open('./data_for_Vae/PGDtest_label.json', 'r', encoding='utf-8') as yrf:
#     test_y = json.load(yrf)
with open('../data_for_Vae/PGDtest_data.json', 'r', encoding='utf-8') as yrf:
    test_pgd = json.load(yrf)
with open('../data_for_Vae/PGDtest_label.json', 'r', encoding='utf-8') as yrf:
    test_y = json.load(yrf)
test_pgd = torch.tensor(test_pgd)
test_x = torch.tensor(test_x)
test_y = torch.tensor(test_y, dtype=torch.int64)
test_set = TensorDataset(test_pgd, test_x, test_y)
# print(test_pgd.shape, test_x.shape, test_y.shape)

train_loader = DataLoader(
    dataset=train_set,
    batch_size=128,
    shuffle=True,
    num_workers=1
)
test_loader = DataLoader(
    dataset=test_set,
    batch_size=256,
    shuffle=False,
    num_workers=1
)

