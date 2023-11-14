import numpy as np
import torch.utils.data as Data
from advertorch.attacks.decoupled_direction_norm import DDNL2Attack
import torch
import os
from AAADefense_mode import VAE
import torchvision
import torchvision.transforms as transforms
from attack_model import victim

device = 'cuda'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

defense_mode = VAE()
defense_mode = defense_mode.to(device)
defense_mode.load_state_dict(torch.load("../mode_save/CVAE-GAN-VAE.pth"))
defense_mode.eval()


test_data = torchvision.datasets.MNIST(
    root='../data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
)
test_loader = Data.DataLoader(test_data, batch_size=1, num_workers=0, shuffle=False)


victim_mode = victim().to(device)
victim_mode.load_state_dict(torch.load("../LeNet_5.pth"))
victim_mode.eval()

defense_mode = VAE()
defense_mode = defense_mode.to(device)
defense_mode.load_state_dict(torch.load("../mode_save/CVAE-GAN-VAE.pth"))
defense_mode.eval()


count = 0
total = 0
adversary = DDNL2Attack(victim_mode, init_norm=2.)

for x, y in test_loader:
    x, y = x.to(device), y.to(device)
    _, pre1 = victim_mode(x).max(1)
    if pre1 == y:
        x_ddn = adversary.perturb(x, y=None)
        # PGD non-target
        print('---')
        _, y_pred_ddn = victim_mode(x_ddn).max(1)  # model prediction on FGM adversarial examples
        if y_pred_ddn.eq(y).sum().item() == 0:
            output = defense_mode.rec_smi(x_ddn)
            _, pre = victim_mode(output).max(1)
            count += pre.eq(y).sum().item()
            total += 1
print(count, total, count / total)
