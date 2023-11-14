import numpy as np
import torch.utils.data as Data
import torch
from AAADefense_mode import VAE
import os
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
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

count = 0
total = 0

for x, y in test_loader:
    x, y = x.to(device), y.to(device)
    t = torch.tensor([4]).to(device)
    if y != t:
        _, pre1 = victim_mode(x).max(1)
        if pre1 == y:
            x_pgd_t = projected_gradient_descent(victim_mode, x, 0.3, 0.01, 40, np.inf, y=t, targeted=True)
            # PGD target
            print('---')
            _, y_pred_pgd = victim_mode(x_pgd_t).max(1)  # model prediction on FGM adversarial examples
            if y_pred_pgd.eq(y).sum().item() == 0:
                output = defense_mode.rec_smi(x_pgd_t)
                _, pre = victim_mode(output).max(1)
                count += pre.eq(y).sum().item()
                total += 1
print("t=4", count, total, 1-(count / total))


