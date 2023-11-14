import torch.optim as optim
import time
import torch.utils.data as Data
import torchvision
import torch.nn as nn
import torch
from torch.optim import SGD
from torchvision import transforms
import os

from torch.nn import Module
from torch import nn


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
train_data_loader = Data.DataLoader(train_data, batch_size=256, num_workers=3, shuffle=True)
test_data_loader = Data.DataLoader(test_data, batch_size=256, num_workers=0, shuffle=True)


device = 'cuda'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
net = LeNet5()
net.load_state_dict(torch.load("LeNet_5.pth"))
net.eval()
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)


# Training
def train(epoch):
    print('Epoch {}/{}'.format(epoch + 1, 1000))
    print('-' * 10)
    start_time = time.time()
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_data_loader):
        # print(inputs.shape)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    end_time = time.time()
    print('TrainLoss: %.3f | TrainAcc: %.3f%% (%d/%d) | Time Elapsed %.3f sec' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, end_time-start_time))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    print(acc)
    if acc == 99.36:
        print('Saving..')
        torch.save(net.state_dict(), "LeNet_5.pth")
        best_acc = acc

for epoch in range(start_epoch, start_epoch+1000):
    train(epoch)
    test(epoch)
print(best_acc)