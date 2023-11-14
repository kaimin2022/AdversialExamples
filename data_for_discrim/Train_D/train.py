import json
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import torchvision
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.layer1 = nn.Linear(200, 2)

    def forward(self, x):
        x = self.layer1(x)
        return x

# ---------------------get test -----------------------------

with open('../Train_data_dis.json', 'r', encoding='utf-8') as yrf:
    Train_data = json.load(yrf)
with open('../Train_label_dis', 'r', encoding='utf-8') as yrf:
    Train_label = json.load(yrf)

train_x = torch.tensor(Train_data)
train_y = torch.tensor(Train_label, dtype=torch.int64)
train_set = TensorDataset(train_x, train_y)
train_loader = DataLoader(
    dataset=train_set,
    batch_size=64,
    shuffle=True,
    num_workers=0

)

with open('../Test_data.json', 'r', encoding='utf-8') as yrf:
    test_data = json.load(yrf)
with open('../Test_label', 'r', encoding='utf-8') as yrf:
    test_label = json.load(yrf)

test_x = torch.tensor(test_data)
test_y = torch.tensor(test_label, dtype=torch.int64)
test_set = TensorDataset(test_x, test_y)
test_loader = DataLoader(
    dataset=test_set,
    batch_size=64,
    shuffle=True,
    num_workers=0

)
# ----------------train and test-----------------
mode = LR()
mode = mode.to(device)
optimizer = optim.Adam(mode.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
criterion = nn.CrossEntropyLoss()


# Training
def train(epoch):
    print('Epoch {}/{}'.format(epoch + 1, 200))
    print('-' * 10)
    start_time = time.time()
    mode.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = mode(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    end_time = time.time()
    print('TrainLoss: %.3f | TrainAcc: %.3f%% (%d/%d) | Time Elapsed %.3f sec' % (
    train_loss / (batch_idx + 1), 100. * correct / total, correct, total, end_time - start_time))


def test(epoch):
    global best_acc
    mode.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = mode(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print('TestAcc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        torch.save(mode.state_dict(), "Dis_AE(x).pth")
        best_acc = acc

for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
print(best_acc)
