import json
import os
import torch
import torch.utils.data as Data
import numpy as np
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
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
        self.linear2 = nn.Linear(200, 84)
        self.relu4 = nn.ReLU(inplace=True)
        self.linear3 = nn.Linear(84, 10)

    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.linear1(out))
        out = self.relu4(self.linear2(out))
        out = self.linear3(out)
        return out

train_data = torchvision.datasets.MNIST(
    root='../data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
)
test_data = torchvision.datasets.MNIST(
    root='../data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
)
train_data_loader = Data.DataLoader(train_data, batch_size=1, num_workers=0, shuffle=False)
test_data_loader = Data.DataLoader(test_data, batch_size=1, num_workers=0, shuffle=False)

device = 'cuda'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
net = LeNet5()
net.load_state_dict(torch.load("../LeNet_5.pth"))
net = net.to(device)

x_trian_pgd = []
y_train_pgd = []
x_test_pgd = []
y_test_pgd = []
count = 0
for index, (x, y) in enumerate(train_data_loader):
    x = x.to(device)
    y = y.to(device)
    net.eval()
    print("get train_pgd")
    x_pgd = projected_gradient_descent(net, x, 0.3, 0.01, 40, np.inf)
    # PGD non-target
    x_pgd = x_pgd.squeeze(dim=0)
    # print(x_pgd.shape)
    y = y.squeeze()
    x_pgd = x_pgd.cpu()
    y = y.cpu()
    x_pgd = x_pgd.detach().numpy().tolist()
    y = y.detach().numpy().tolist()
    x_trian_pgd.append(x_pgd)
    y_train_pgd.append(y)
with open('PGDtrain_data.json', 'w', encoding='utf-8') as xjf:
    json.dump(x_trian_pgd, xjf)
with open('PGDtrain_label.json', 'w', encoding='utf-8') as yjf:
    json.dump(y_train_pgd, yjf)


for index, (x, y) in enumerate(test_data_loader):
    # if index == 16:
    #     break
    x = x.to(device)
    y = y.to(device)
    net.eval()
    print("get test_pgd")
    x_pgd = projected_gradient_descent(net, x, 0.3, 0.01, 40, np.inf)
    x_pgd = x_pgd.squeeze(dim=0)
    y = y.squeeze()
    x_pgd = x_pgd.cpu()
    y = y.cpu()
    x_pgd = x_pgd.detach().numpy().tolist()
    y = y.detach().numpy().tolist()
    x_test_pgd.append(x_pgd)
    y_test_pgd.append(y)

with open('PGDtest_data.json', 'w', encoding='utf-8') as xjf:
    json.dump(x_test_pgd, xjf)
with open('PGDtest_label.json', 'w', encoding='utf-8') as yjf:
    json.dump(y_test_pgd, yjf)

