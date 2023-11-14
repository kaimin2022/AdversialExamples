import json

from AAADefense_mode import VAE
import torch.utils.data as Data
import torch
import os
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
import torchvision
import torchvision.transforms as transforms
from attack_model import victim

test_data = torchvision.datasets.MNIST(
    root='../data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
)
test_loader = Data.DataLoader(test_data, batch_size=1, num_workers=0, shuffle=False)
device = 'cuda'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
victim_mode = victim().to(device)
victim_mode.load_state_dict(torch.load("../LeNet_5.pth"))
victim_mode.eval()

defense_mode = VAE()
defense_mode = defense_mode.to(device)
defense_mode.load_state_dict(torch.load("../mode_save/CVAE-GAN-VAE.pth"))
defense_mode.eval()

total = 0
count = 0


for index,(x, y) in enumerate(test_loader):
    x, y = x.to(device), y.to(device)
    _, pre1 = victim_mode(x).max(1)
    if pre1 == y:
        x_cw = carlini_wagner_l2(victim_mode, x, n_classes=10, confidence=1, initial_const=1, max_iterations=1000)
        print(index)
        _, y_pred_cw = victim_mode(x_cw).max(1)  # model prediction on FGM adversarial examples
        if y_pred_cw.eq(y).sum().item() == 0:
            output = defense_mode.rec_smi(x_cw)
            _, pre = victim_mode(output).max(1)
            count += pre.eq(y).sum().item()
            total += 1
print(count, total, count / total)
