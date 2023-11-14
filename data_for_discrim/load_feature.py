import json
import os
import numpy as np
import re
import torch
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transform

device = 'cuda'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def _load_data(dir):
    # loay x
    files = os.listdir(dir)

    idFiles = [(int(re.match('x_(.*?).npy', fn).group(1)), fn) for fn in files]
    idFiles.sort(key=lambda x: x[0])

    arrayFns = [os.path.join(dir, fn) for _, fn in idFiles]
    arrays = [np.load(fn) for fn in arrayFns]
    x = np.concatenate(arrays, axis=0)

    # load y
    y = np.load('y.npy')
    return x, y


def load_robust_dataset():
    return _load_data('robust')


def load_nonrobust_dataset():
    return _load_data('nonrobust')


robust = load_robust_dataset()
# print(robust[0].shape)  # data (60000, 28, 28)
# print(robust[1].shape)  # label (60000)
data = robust[0]  # Numpy
label = robust[1]
data = torch.from_numpy(data)
label = torch.from_numpy(label)
train_data = Data.TensorDataset(data, label)

# 切分Robust数据集
R_train, R_test = train_test_split(
    train_data, test_size=1 / 6, random_state=123
)
transforms = transform.Compose([
    transform.ToTensor(),
    transform.Normalize(mean=(0.1307,), std=(0.3081,))
])
Robust_transforms = transform.Compose([
    transform.Normalize(mean=(0.1307,), std=(0.3081,))
])
# Natural数据集
Nature_train = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transforms)
Nature_test = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transforms)

Robust_train_loader = DataLoader(  # 50000
    dataset=R_train,
    batch_size=1,
    shuffle=False
)
Robust_test_loader = DataLoader(  # 10000
    dataset=R_test,
    batch_size=1,
    shuffle=False
)
Nature_train_loader = Data.DataLoader(  # 60000
    Nature_train,
    batch_size=1,
    shuffle=True
)
Nature_test_loader = Data.DataLoader(  # 10000
    Nature_test,
    batch_size=1,
    shuffle=True
)

x_train = []
y_train = []
x_test = []
y_test = []
# 组合训练集
for x, y in Robust_train_loader:  # 50000
    x = Robust_transforms(x)
    y = torch.tensor([1])  # 标签改为1，表示是鲁棒特征。
    # x = x.squeeze()
    y = y.squeeze()
    x_train.append(x)
    y_train.append(y)
for x, y in Nature_train_loader:  # 60000

    y = torch.tensor([0])  # 标签改为0， 正常样本特征。
    x = x.squeeze(dim=0)
    y = y.squeeze()
    x_train.append(x)
    y_train.append(y)
train_x = torch.stack(x_train, 0)  # 将具有tensor的list转为tensor
train_y = torch.stack(y_train, 0)
train_set = Data.TensorDataset(train_x, train_y)
# 组合测试集
for x, y in Robust_test_loader:  # 10000
    x = Robust_transforms(x)
    y = torch.tensor([1])
    # x = x.squeeze()
    y = y.squeeze()
    x_test.append(x)
    y_test.append(y)
for x, y in Nature_test_loader:  # 10000 1, 1, 28, 28
    y = torch.tensor([0])
    x = x.squeeze(dim=0)
    y = y.squeeze()
    x_test.append(x)
    y_test.append(y)
test_x = torch.stack(x_test, 0)
test_y = torch.stack(y_test, 0)
test_set = Data.TensorDataset(test_x, test_y)
# 定义数据加载器
train_loader = DataLoader(
    dataset=train_set,  # 64, 1, 28,28
    batch_size=1,
    shuffle=True,
    num_workers=0
)
test_loader = DataLoader(
    dataset=test_set,
    batch_size=1,
    shuffle=True,
    num_workers=0
)
# above is get dataloader for discrim, the data size is 1,28,28
# ----------------------------------------------------------------
# next need put this data to the classify model get latent feature for LR-D
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(7*7*64, 200)
        self.relu3 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(200, 10)

    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        return out
mode = LeNet5().to(device)
mode.load_state_dict(torch.load("../LeNet_5.pth"))
mode.eval()
train_data = []
train_label = []
test_data = []
test_label = []
for index1, (x, y) in enumerate(train_loader):
    x = x.to(device)
    y = y.to(device)
    x_lat = mode(x)
    # print(x_lat.shape)
    x_lat = x_lat.squeeze(dim=0)
    x_lat = x_lat.cpu()
    x_lat = x_lat.detach().numpy().tolist()
    train_data.append(x_lat)
    y = y.squeeze()
    y = y.cpu()
    y = y.detach().numpy().tolist()
    train_label.append(y)
print(index1)
with open('Train_data_dis.json', 'w', encoding='utf-8') as xjf:
    json.dump(train_data, xjf)
with open('Train_label_dis', 'w', encoding='utf-8') as yjf:
    json.dump(train_label, yjf)

for index,(x, y) in enumerate(test_loader):
    x = x.to(device)
    y = y.to(device)
    x_lat = mode(x)
    x_lat = x_lat.squeeze(dim=0)
    x_lat = x_lat.cpu()
    x_lat = x_lat.detach().numpy().tolist()
    test_data.append(x_lat)
    y = y.squeeze()
    y = y.cpu()
    y = y.detach().numpy().tolist()
    test_label.append(y)
print(index)
with open('Test_data.json', 'w', encoding='utf-8') as xjf:
    json.dump(test_data, xjf)
with open('Test_label', 'w', encoding='utf-8') as yjf:
    json.dump(test_label, yjf)

