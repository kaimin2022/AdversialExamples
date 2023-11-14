from advertorch.attacks.jsma import JacobianSaliencyMapAttack
from torch import nn
import torch
import os
import torch.utils.data as Data
import torchvision
from torchvision.transforms import transforms

device = 'cuda'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
nz = 200
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


victim_mode = LeNet5()
victim_mode = victim_mode.to(device)
victim_mode.load_state_dict(torch.load("../LeNet_5.pth"))
victim_mode.eval()

robust_model = LeNet5()
robust_model = robust_model.to(device)
robust_model.load_state_dict(torch.load("./Para/R_LeNet_5.pth"))
robust_model.eval()

test_data = torchvision.datasets.MNIST(
    root='../data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
)
val_loader = Data.DataLoader(test_data, batch_size=1024, num_workers=0, shuffle=False)
count = 0
total = 0


adversary = JacobianSaliencyMapAttack(victim_mode, num_classes=10)

count = 0
total = 0


for x, y in val_loader:
    x, y = x.to(device), y.to(device)
    t = torch.tensor([6]).to(device)
    t = t.to(device)
    if y != t:
        _, pre1 = victim_mode(x).max(1)
        if pre1 == y:
            x_jsma = adversary.perturb(x, t)
            print('---')
            _, y_pred_jsma = victim_mode(x_jsma).max(1)  # model prediction on FGM adversarial examples
            if y_pred_jsma.eq(pre1).sum().item() == 0:
                output = defense_mode.rec_smi(x_jsma)
                _, pre = victim_mode(output).max(1)
                count += pre.eq(y).sum().item()
                total += 1
print("T=6", count, total, count / total)


